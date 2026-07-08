"""High-level interface in the spirit of scikit-learn / SetFit: new, fit, save, load, predict."""
import math
import time

import torch
import torch.nn.functional as F

from .data import Vocab, make_batch, split_pairs
from .model import GeoConfig, GeoModel, pick_device


class GeoLM:
    """A narrow geometric model for one task.

    Example:
        model = GeoLM.new(train_pairs, dim=128, hidden=512)
        model.fit(train_pairs, steps=20000)
        model.save("task.geolm")

        model = GeoLM.load("task.geolm")
        model.predict("357+468=")   # "825"
    """

    def __init__(self, model: GeoModel, vocab: Vocab, device="auto"):
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.vocab = vocab

    # ---------- creation / persistence ----------
    @classmethod
    def new(cls, pairs, dim=128, hidden=512, layers=2, dropout=0.1, max_new=64, device="auto"):
        """Creates an untrained model; the vocabulary is built from the task data."""
        vocab = Vocab.build(pairs)
        cfg = GeoConfig(vocab_size=len(vocab), dim=dim, hidden=hidden,
                        layers=layers, dropout=dropout, max_new=max_new)
        return cls(GeoModel(cfg), vocab, device)

    def save(self, path):
        """A single file: configuration plus vocabulary plus weights."""
        torch.save({
            "format": "geolm-v1",
            "config": self.model.cfg.to_dict(),
            "vocab": self.vocab.tokens,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }, path)
        return path

    @classmethod
    def load(cls, path, device="auto"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob.get("format") == "geolm-v1", "not a geolm file"
        model = GeoModel(GeoConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        return cls(model, Vocab(blob["vocab"]), device)

    # ---------- training ----------
    def fit(self, pairs, steps=20000, batch_size=256, lr=1e-3, warmup=None,
            val_frac=0.05, eval_every=None, log=print, seed=0):
        """Trains on (input, output) pairs. Returns the best held-out exact match."""
        import random
        rng = random.Random(seed)
        train, val = split_pairs(pairs, val_frac=val_frac, seed=seed)
        warmup = warmup or max(100, steps // 20)
        eval_every = eval_every or max(200, steps // 10)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        best = -1.0
        best_state = None
        t0 = time.time()
        self.model.train()
        for step in range(1, steps + 1):
            frac = step / warmup if step < warmup else 1.0
            cos = 0.1 + 0.45 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, steps - warmup)))
            for g in opt.param_groups:
                g["lr"] = lr * min(frac, cos)
            batch = [train[rng.randrange(len(train))] for _ in range(min(batch_size, len(train)))]
            x, y = make_batch(batch, self.vocab, self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            if step % eval_every == 0 or step == steps:
                acc = self.evaluate(val)
                mark = ""
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    mark = "  <- best"
                log(f"[geolm] step {step}/{steps}  loss={loss.item():.3f}  "
                    f"exact match={acc:.1f}%{mark}  ({time.time()-t0:.0f}s)")
        if best_state is not None:
            self.model.load_state_dict(best_state)   # keep the best checkpoint
        return best

    # ---------- evaluation / inference ----------
    @torch.no_grad()
    def evaluate(self, pairs, batch_size=256):
        """Share of exact answer matches, in percent."""
        self.model.eval()
        hit = 0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            preds = self.predict_batch([inp for inp, _ in chunk])
            hit += sum(1 for p, (_, out) in zip(preds, chunk) if p == out)
        self.model.train()
        return hit / max(1, len(pairs)) * 100

    @torch.no_grad()
    def predict(self, text: str, temperature: float = 0.0) -> str:
        """The model's answer for a single input."""
        return self.predict_batch([text], temperature=temperature)[0]

    @torch.no_grad()
    def predict_batch(self, texts, temperature: float = 0.0):
        """Answers for a list of inputs in one pass (fast).

        Inputs of different lengths are handled with packed sequences so that
        padding never flows through the recurrent engine and cannot corrupt
        its state.
        """
        self.model.eval()
        vocab, device = self.vocab, self.device
        enc = [vocab.encode(t) + [vocab.sep_id] for t in texts]
        lengths = torch.tensor([len(e) for e in enc])
        width = int(lengths.max())
        x = torch.full((len(enc), width), vocab.pad_id, dtype=torch.long, device=device)
        for r, e in enumerate(enc):
            x[r, : len(e)] = torch.tensor(e, device=device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            self.model.points(x), lengths, batch_first=True, enforce_sorted=False)
        _, state = self.model.rnn(packed)
        cur = self.model.proj(state[-1])   # top-layer output at the last REAL character
        done = torch.zeros(len(enc), dtype=torch.bool, device=device)
        outs = [[] for _ in enc]
        for _ in range(self.model.cfg.max_new):
            logits = self.model.logits(cur)
            if temperature and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            for r in range(len(enc)):
                if not done[r]:
                    if int(nxt[r]) == vocab.eos_id:
                        done[r] = True
                    else:
                        outs[r].append(int(nxt[r]))
            if bool(done.all()):
                break
            feats, state = self.model.trajectory(nxt.unsqueeze(1), state)
            cur = feats[:, -1]
        return [vocab.decode(o) for o in outs]
