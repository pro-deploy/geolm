"""Vocabulary and data. Data format is JSONL: {"input": "...", "output": "..."} per line."""
import json
import random

import torch

PAD, SEP, EOS = "<pad>", "<sep>", "<eos>"
SPECIALS = [PAD, SEP, EOS]


class Vocab:
    """Character-level vocabulary built from task data. Stored inside the model file."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def build(cls, pairs) -> "Vocab":
        chars = set()
        for inp, out in pairs:
            chars.update(inp)
            chars.update(out)
        return cls(SPECIALS + sorted(chars))

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_id(self):
        return self.index[PAD]

    @property
    def sep_id(self):
        return self.index[SEP]

    @property
    def eos_id(self):
        return self.index[EOS]

    def encode(self, text: str):
        unk = self.index.get(" ", self.pad_id)
        return [self.index.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        stop = {self.pad_id, self.sep_id, self.eos_id}
        return "".join(self.tokens[i] for i in ids if i not in stop)


def read_jsonl(path):
    """Reads JSONL with input/output fields into a list of (input, output) pairs."""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((str(row["input"]), str(row["output"])))
    return pairs


def split_pairs(pairs, val_frac=0.05, seed=0):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac)) if len(pairs) > 20 else max(1, len(pairs) // 5)
    return pairs[n_val:], pairs[:n_val]


def make_batch(pairs, vocab: Vocab, device):
    """Builds a batch: input <sep> output <eos>; the loss covers only the output part.

    Returns x [B, T-1] and y [B, T-1], where input positions and padding are marked
    with -100 and excluded from training (standard loss masking).
    """
    rows = []
    for inp, out in pairs:
        ids = vocab.encode(inp) + [vocab.sep_id] + vocab.encode(out) + [vocab.eos_id]
        rows.append((ids, len(vocab.encode(inp)) + 1))
    width = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), width - 1), vocab.pad_id, dtype=torch.long)
    y = torch.full((len(rows), width - 1), -100, dtype=torch.long)
    for r, (ids, prompt_len) in enumerate(rows):
        seq = torch.tensor(ids, dtype=torch.long)
        x[r, : len(ids) - 1] = seq[:-1]
        tgt = seq[1:].clone()
        tgt[: prompt_len - 1] = -100          # do not train on the input, only on the answer
        y[r, : len(ids) - 1] = tgt
    return x.to(device), y.to(device)
