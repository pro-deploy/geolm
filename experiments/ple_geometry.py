"""Experiment: per-layer token embeddings (PLE, as popularized by Gemma 3n and
discussed in the habrGPT article) adapted to the geometric architecture.

Honest note: with a character-level vocabulary the capacity effect of PLE is
negligible (the tables are tiny). What remains testable is per-layer token
injection: every block sees the raw token identity directly instead of through
the residual stream. We measure whether that helps a deep residual stack.

Task: the full mixed ten-operation calculator (the earlier benchmark where the
base architecture reached 91.5% overall, held back mostly by division).

Variants, all about 2.6M parameters, same data and budget:
  1. base        stacked GRU (geolm today)
  2. deep        residual pre-norm GRU blocks
  3. deep+ple    residual blocks, each block also receives its own per-layer
                 embedding of the current input token
"""
import math
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

# ---------------- vocabulary ----------------
CHARS = "0123456789.+-*/^()=qscle"
PAD, EOS = 0, 1
c2i = {c: i + 2 for i, c in enumerate(CHARS)}
i2c = {i + 2: c for i, c in enumerate(CHARS)}
V = len(CHARS) + 2


def encode(s):
    return [c2i[ch] for ch in s]


def decode(ids):
    return "".join(i2c.get(i, "") for i in ids)


# ---------------- task: mixed ten-operation calculator ----------------
def sample():
    op = random.choice(list("+-*/^qscle"))
    if op in "+-":
        a, b = random.randint(0, 999), random.randint(0, 999)
        e, v = f"{a}{op}{b}", (a + b if op == "+" else a - b)
    elif op == "*":
        a, b = random.randint(0, 99), random.randint(0, 99)
        e, v = f"{a}*{b}", a * b
    elif op == "/":
        a, b = random.randint(0, 999), random.randint(1, 99)
        e, v = f"{a}/{b}", a / b
    elif op == "^":
        a, b = random.randint(0, 12), random.randint(0, 3)
        e, v = f"{a}^{b}", a ** b
    elif op == "q":
        x = random.randint(0, 9999) / 100
        e, v = f"q({x:.2f})", math.sqrt(x)
    elif op == "s":
        x = random.randint(-628, 628) / 100
        e, v = f"s({x:.2f})", math.sin(x)
    elif op == "c":
        x = random.randint(-628, 628) / 100
        e, v = f"c({x:.2f})", math.cos(x)
    elif op == "l":
        x = random.randint(1, 9999) / 100
        e, v = f"l({x:.2f})", math.log(x)
    else:
        x = random.randint(-400, 600) / 100
        e, v = f"e({x:.2f})", math.exp(x)
    return op, f"{e}=", f"{v:.4f}"


def make_batch(n):
    rows = [sample() for _ in range(n)]
    seqs = [encode(i) + encode(o) + [EOS] for _, i, o in rows]
    plens = [len(encode(i)) for _, i, _ in rows]
    width = max(len(s) for s in seqs)
    x = torch.full((n, width - 1), PAD, dtype=torch.long)
    y = torch.full((n, width - 1), -100, dtype=torch.long)
    for r, (s, pl) in enumerate(zip(seqs, plens)):
        t = torch.tensor(s)
        x[r, : len(s) - 1] = t[:-1]
        tgt = t[1:].clone()
        tgt[: pl - 1] = -100
        y[r, : len(s) - 1] = tgt
    return x.to(dev), y.to(dev)


# ---------------- models ----------------
class GeoOut(nn.Module):
    """Shared point cloud plus the nearest-point output head."""

    def __init__(s, D):
        super().__init__()
        s.points = nn.Embedding(V, D)
        nn.init.normal_(s.points.weight, std=0.1)
        s.ls = nn.Parameter(torch.zeros(1))

    def logits(s, f):
        E = s.points.weight
        return -((f * f).sum(-1, keepdim=True) - 2 * f @ E.t() + (E * E).sum(-1)) * torch.exp(s.ls)


class StackGeo(nn.Module):
    def __init__(s, D=128, H=512, L=2):
        super().__init__()
        s.geo = GeoOut(D)
        s.gru = nn.GRU(D, H, num_layers=L, batch_first=True, dropout=0.1)
        s.proj = nn.Linear(H, D)

    def forward(s, x):
        out, _ = s.gru(s.geo.points(x))
        return s.geo.logits(s.proj(out))

    def prefix(s, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            s.geo.points(x), lengths, batch_first=True, enforce_sorted=False)
        _, state = s.gru(packed)
        return s.proj(state[-1]), state

    def step(s, nxt, state):
        out, state = s.gru(s.geo.points(nxt).unsqueeze(1), state)
        return s.proj(out[:, 0]), state


class ResBlock(nn.Module):
    def __init__(s, H, ple: bool):
        super().__init__()
        s.ln = nn.LayerNorm(H)
        s.gru = nn.GRU(H, H, batch_first=True)
        s.ple = nn.Embedding(V, H) if ple else None
        if s.ple is not None:
            nn.init.normal_(s.ple.weight, std=0.02)

    def inject(s, h, tokens):
        return h + s.ple(tokens) if s.ple is not None else h

    def forward(s, h, tokens):
        out, _ = s.gru(s.ln(s.inject(h, tokens)))
        return h + out


class DeepGeo(nn.Module):
    def __init__(s, D=128, H=320, blocks=4, ple=False):
        super().__init__()
        s.geo = GeoOut(D)
        s.inmap = nn.Linear(D, H)
        s.blocks = nn.ModuleList([ResBlock(H, ple) for _ in range(blocks)])
        s.outmap = nn.Linear(H, D)

    def forward(s, x):
        h = s.inmap(s.geo.points(x))
        for blk in s.blocks:
            h = blk(h, x)
        return s.geo.logits(s.outmap(h))

    def prefix(s, x, lengths):
        T = x.shape[1]
        h = s.inmap(s.geo.points(x))
        states = []
        for blk in s.blocks:
            packed = nn.utils.rnn.pack_padded_sequence(
                blk.ln(blk.inject(h, x)), lengths, batch_first=True, enforce_sorted=False)
            out_p, st = blk.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True, total_length=T)
            h = h + out
            states.append(st)
        idx = (lengths - 1).to(dev)
        last = h[torch.arange(h.shape[0], device=dev), idx]
        return s.outmap(last), states

    def step(s, nxt, states):
        tok = nxt.unsqueeze(1)
        h = s.inmap(s.geo.points(tok))
        new_states = []
        for blk, st in zip(s.blocks, states):
            out, st2 = blk.gru(blk.ln(blk.inject(h, tok)), st)
            h = h + out
            new_states.append(st2)
        return s.outmap(h[:, 0]), new_states


# ---------------- training and evaluation ----------------
def train(model, steps=20000, bs=192, lr=1e-3, warm=600, tag=""):
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    for st in range(1, steps + 1):
        cur = lr * st / warm if st < warm else lr * (0.1 + 0.45 * (1 + math.cos(
            math.pi * (st - warm) / (steps - warm))))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = make_batch(bs)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % 4000 == 0:
            q = evaluate(model, n=1000)
            print(f"   [{tag}] step {st}/{steps} loss={loss.item():.3f} "
                  f"tol={q['overall']:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, n=5000, bs=500):
    model.eval()
    hit = defaultdict(lambda: [0, 0])
    for i in range(0, n, bs):
        rows = [sample() for _ in range(min(bs, n - i))]
        enc = [encode(inp) for _, inp, _ in rows]
        lengths = torch.tensor([len(e) for e in enc])
        width = int(lengths.max())
        x = torch.full((len(enc), width), PAD, dtype=torch.long, device=dev)
        for r, e in enumerate(enc):
            x[r, : len(e)] = torch.tensor(e, device=dev)
        cur, state = model.prefix(x, lengths)
        done = torch.zeros(len(enc), dtype=torch.bool, device=dev)
        outs = [[] for _ in enc]
        for _ in range(12):
            nxt = model.geo.logits(cur).argmax(-1) if hasattr(model, "geo") else None
            nxt = model.geo.logits(cur).argmax(-1)
            for r in range(len(enc)):
                if not done[r]:
                    if int(nxt[r]) == EOS:
                        done[r] = True
                    else:
                        outs[r].append(int(nxt[r]))
            if bool(done.all()):
                break
            cur, state = model.step(nxt, state)
        for r, (op, _, ans) in enumerate(rows):
            hit[op][1] += 1
            try:
                if abs(float(decode(outs[r])) - float(ans)) < 0.01:
                    hit[op][0] += 1
            except ValueError:
                pass
    model.train()
    res = {op: h[0] / h[1] * 100 for op, h in hit.items() if h[1]}
    res["overall"] = sum(h[0] for h in hit.values()) / max(1, sum(h[1] for h in hit.values())) * 100
    return res


def params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    variants = [
        ("base", StackGeo()),
        ("deep", DeepGeo(ple=False)),
        ("deep+ple", DeepGeo(ple=True)),
    ]
    results = {}
    for tag, model in variants:
        print(f"=== {tag}: {params_m(model):.2f}M parameters ===", flush=True)
        model = train(model, tag=tag)
        results[tag] = evaluate(model, n=5000)
    names = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow",
             "q": "sqrt", "s": "sin", "c": "cos", "l": "log", "e": "exp"}
    print("\n========== MIXED CALCULATOR, within 0.01 (5000 held-out) ==========", flush=True)
    header = f"{'op':6}" + "".join(f"{t:>10}" for t, _ in variants)
    print(header, flush=True)
    for op in "+-*/^qscle":
        row = f"{names[op]:6}" + "".join(f"{results[t].get(op, 0):>9.1f}%" for t, _ in variants)
        print(row, flush=True)
    print(f"{'ALL':6}" + "".join(f"{results[t]['overall']:>9.1f}%" for t, _ in variants), flush=True)


if __name__ == "__main__":
    main()
