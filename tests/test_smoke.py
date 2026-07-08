"""Quick health check: train a tiny model to reverse strings.

Run: python tests/test_smoke.py  (or pytest tests/)
Takes about a minute on a laptop CPU.
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geolm import GeoLM  # noqa: E402


def make_pairs(n=4000, width=6):
    rng = random.Random(0)
    pairs = []
    for _ in range(n):
        s = "".join(rng.choice("abcde") for _ in range(width))
        pairs.append((s + "=", s[::-1]))
    return pairs


def test_train_save_load_predict():
    pairs = make_pairs()
    model = GeoLM.new(pairs, dim=48, hidden=160, layers=1, max_new=10, device="cpu")
    best = model.fit(pairs, steps=900, batch_size=128, eval_every=300, log=lambda *_: None)
    assert best > 60, f"model failed to learn: {best:.1f}%"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reverse.geolm")
        model.save(path)
        loaded = GeoLM.load(path, device="cpu")
        assert loaded.predict("abcde=") == model.predict("abcde=")

    sample = pairs[:200]
    acc = model.evaluate(sample)
    assert acc > 60, f"low accuracy after loading: {acc:.1f}%"
    print(f"OK: training works, accuracy {acc:.1f}%, save/load/inference intact")


if __name__ == "__main__":
    test_train_save_load_predict()
