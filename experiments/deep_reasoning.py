"""Experiment: transformer-style depth (residual blocks) and scratchpad reasoning.

Task: division a/b to 4 decimal places, the known failure case of the calculator
benchmark (about 15% for both geolm and the transformer at 2.6M parameters).

Four variants, all about the same parameter count, same data and budget:
  1. base+direct    stacked GRU (as in geolm today), answer directly
  2. deep+direct    residual pre-norm GRU blocks (transformer-style stacking)
  3. base+reason    stacked GRU, output = long-division scratchpad, then answer
  4. deep+reason    residual blocks plus the scratchpad

Metrics: exact final answer, answer within 0.01, average generated characters,
batched inference latency per query.
"""
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

# ---------------- vocabulary ----------------
CHARS = "0123456789./=|r"
PAD, EOS = 0, 1
c2i = {c: i + 2 for i, c in enumerate(CHARS)}
i2c = {i + 2: c for i, c in enumerate(CHARS)}
V = len(CHARS) + 2


def encode(s):
    return [c2i[ch] for ch in s]


def decode(ids):
    return "".join(i2c.get(i, "") for i in ids)


# ---------------- task: division with an optional scratchpad ----------------
def sample(scratch: bool):
    a, b = random.randint(0, 999), random.randint(1, 99)
    qint, r = divmod(a, b)
    digs, steps = [], [f"{qint}r{r}"]
    for _ in range(4):
        r *= 10
        d, r = divmod(r, b)
        digs.append(str(d))
        steps.append(f"{d}r{r}")
    ans = f"{qint}.{''.join(digs)}"          # truncated long-division ground truth
    inp = f"{a}/{b}="
    out = ("|".join(steps) + "=" + ans) if scratch else ans
    return inp, out, ans


def make_batch(n, scratch):
    rows = [sample(scratch) for _ in range(n)]
    seqs = [encode(i) + encode(o) + [EOS] for i, o, _ in rows]
    plens = [len(encode(i)) for i, _, _ in rows]
    width = max(len(s) for s in seqs)
    x = torch.full((n, width - 1), PAD, dtype=torch.long)
    y = torch.full((n, width - 1), -100, dtype=torch.long)
    for r, (s, pl) in enumerate(zip(seqs, plens)):
        t = torch.tensor(s)
        x[r, : len(s) - 1] = t[:-1]
        tgt = t[1:].clone()
        tgt[: pl - 1] = -100                  # learn only the output part
        y[r, : len(s) - 1] = tgt
    return x.to(dev), y.to(dev)


# ---------------- model 1: stacked GRU (geolm today) ----------------
class StackGeo(nn.Module):
    def __init__(s, D=128, H=512, L=2):
        super().__init__()
        s.points = nn.Embedding(V, D)
        nn.init.normal_(s.points.weight, std=0.1)
        s.gru = nn.GRU(D, H, num_layers=L, batch_first=True, dropout=0.1)
        s.proj = nn.Linear(H, D)
        s.ls = nn.Parameter(torch.zeros(1))

    def logits(s, f):
        E = s.points.weight
        return -((f * f).sum(-1, keepdim=True) - 2 * f @ E.t() + (E * E).sum(-1)) * torch.exp(s.ls)

    def forward(s, x):
        out, _ = s.gru(s.points(x))
        return s.logits(s.proj(out))

    def prefix(s, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            s.points(x), lengths, batch_first=True, enforce_sorted=False)
        _, state = s.gru(packed)
        return s.proj(state[-1]), state

    def step(s, nxt, state):
        out, state = s.gru(s.points(nxt).unsqueeze(1), state)
        return s.proj(out[:, 0]), state


# ---------------- model 2: residual pre-norm blocks (transformer-style depth) ----------------
class ResBlock(nn.Module):
    """The transformer lesson applied to a GRU engine: layers refine a shared
    residual stream instead of re-digesting it, so intermediate values flow
    untouched past every layer."""

    def __init__(s, H):
        super().__init__()
        s.ln = nn.LayerNorm(H)
        s.gru = nn.GRU(H, H, batch_first=True)

    def forward(s, h):
        out, _ = s.gru(s.ln(h))
        return h + out


class DeepGeo(nn.Module):
    def __init__(s, D=128, H=320, blocks=4):
        super().__init__()
        s.points = nn.Embedding(V, D)
        nn.init.normal_(s.points.weight, std=0.1)
        s.inmap = nn.Linear(D, H)
        s.blocks = nn.ModuleList([ResBlock(H) for _ in range(blocks)])
        s.outmap = nn.Linear(H, D)
        s.ls = nn.Parameter(torch.zeros(1))

    def logits(s, f):
        E = s.points.weight
        return -((f * f).sum(-1, keepdim=True) - 2 * f @ E.t() + (E * E).sum(-1)) * torch.exp(s.ls)

    def forward(s, x):
        h = s.inmap(s.points(x))
        for blk in s.blocks:
            h = blk(h)
        return s.logits(s.outmap(h))

    def prefix(s, x, lengths):
        T = x.shape[1]
        h = s.inmap(s.points(x))
        states = []
        for blk in s.blocks:
            packed = nn.utils.rnn.pack_padded_sequence(
                blk.ln(h), lengths, batch_first=True, enforce_sorted=False)
            out_p, st = blk.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True, total_length=T)
            h = h + out
            states.append(st)
        idx = (lengths - 1).to(dev)
        last = h[torch.arange(h.shape[0], device=dev), idx]
        return s.outmap(last), states

    def step(s, nxt, states):
        h = s.inmap(s.points(nxt)).unsqueeze(1)
        new_states = []
        for blk, st in zip(s.blocks, states):
            out, st2 = blk.gru(blk.ln(h), st)
            h = h + out
            new_states.append(st2)
        return s.outmap(h[:, 0]), new_states


# ---------------- training and evaluation ----------------
def train(model, scratch, steps=12000, bs=192, lr=1e-3, warm=400, tag=""):
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    for st in range(1, steps + 1):
        cur = lr * st / warm if st < warm else lr * (0.1 + 0.45 * (1 + math.cos(
            math.pi * (st - warm) / (steps - warm))))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = make_batch(bs, scratch)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % 3000 == 0:
            q = evaluate(model, scratch, n=500)
            print(f"   [{tag}] step {st}/{steps} loss={loss.item():.3f} "
                  f"exact={q['exact']:.1f}% tol={q['tol']:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, scratch, n=2000, bs=500):
    model.eval()
    max_new = 45 if scratch else 10
    stats = {"exact": 0, "tol": 0, "chars": 0.0}
    t_total = 0.0
    for i in range(0, n, bs):
        rows = [sample(scratch) for _ in range(min(bs, n - i))]
        enc = [encode(inp) for inp, _, _ in rows]
        lengths = torch.tensor([len(e) for e in enc])
        width = int(lengths.max())
        x = torch.full((len(enc), width), PAD, dtype=torch.long, device=dev)
        for r, e in enumerate(enc):
            x[r, : len(e)] = torch.tensor(e, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        cur, state = model.prefix(x, lengths)
        done = torch.zeros(len(enc), dtype=torch.bool, device=dev)
        outs = [[] for _ in enc]
        for _ in range(max_new):
            nxt = model.logits(cur).argmax(-1)
            for r in range(len(enc)):
                if not done[r]:
                    if int(nxt[r]) == EOS:
                        done[r] = True
                    else:
                        outs[r].append(int(nxt[r]))
            if bool(done.all()):
                break
            cur, state = model.step(nxt, state)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_total += time.time() - t0
        for r, (_, _, ans) in enumerate(rows):
            text = decode(outs[r])
            final = text.split("=")[-1]
            stats["chars"] += len(text)
            if final == ans:
                stats["exact"] += 1
            try:
                if abs(float(final) - float(ans)) < 0.01:
                    stats["tol"] += 1
            except ValueError:
                pass
    model.train()
    return {"exact": stats["exact"] / n * 100, "tol": stats["tol"] / n * 100,
            "chars": stats["chars"] / n, "ms": t_total / n * 1000}


def params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    variants = [
        ("base+direct", StackGeo(), False),
        ("deep+direct", DeepGeo(), False),
        ("base+reason", StackGeo(), True),
        ("deep+reason", DeepGeo(), True),
    ]
    results = {}
    for tag, model, scratch in variants:
        print(f"=== {tag}: {params_m(model):.2f}M parameters, scratchpad={scratch} ===", flush=True)
        model = train(model, scratch, tag=tag)
        results[tag] = evaluate(model, scratch, n=2000)
    print("\n================ DIVISION BENCHMARK (2000 held-out) ================", flush=True)
    print(f"{'variant':14}{'exact':>9}{'within 0.01':>13}{'avg chars':>11}{'ms/query':>10}", flush=True)
    for tag, q in results.items():
        print(f"{tag:14}{q['exact']:>8.1f}%{q['tol']:>12.1f}%{q['chars']:>11.1f}{q['ms']:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
