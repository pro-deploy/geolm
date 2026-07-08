"""End-to-end example: generate data, train, check answers, save the model.

Run:   python examples/calculator.py
Then:  geolm predict --model calc.geolm --input "357+468="
"""
import random

from geolm import GeoLM

random.seed(0)


def make_pairs(n=60000):
    """Addition and subtraction up to 999: a narrow task with exact answers."""
    pairs = []
    for _ in range(n):
        a, b = random.randint(0, 999), random.randint(0, 999)
        if random.random() < 0.5:
            pairs.append((f"{a}+{b}=", str(a + b)))
        else:
            pairs.append((f"{a}-{b}=", str(a - b)))
    return pairs


def main():
    pairs = make_pairs()
    model = GeoLM.new(pairs, dim=96, hidden=384, layers=2, max_new=8)
    n_par = sum(p.numel() for p in model.model.parameters())
    print(f"parameters: {n_par/1e6:.2f}M; examples: {len(pairs)}")

    model.fit(pairs, steps=6000, batch_size=256)

    for q in ["357+468=", "999-1=", "12+34=", "500-501="]:
        print(f"  {q}{model.predict(q)}")

    path = model.save("calc.geolm")
    print(f"model saved: {path}")
    print("try it: geolm predict --model calc.geolm --input '357+468='")


if __name__ == "__main__":
    main()
