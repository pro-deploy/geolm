# Testing guide

This guide is written for someone without a machine-learning background. Every step
gives a command, the expected result, and what passing that step proves. Run all
commands from the `geolm` project folder. Russian version: [TESTING.ru.md](TESTING.ru.md).

## Level 0. Setup

```bash
pip install -e .
python3 -c "import geolm; print('geolm', geolm.__version__)"
```

Expected: `geolm 0.1.0`. Proves the library installs and imports. If the `geolm`
command is not found afterwards, add your user scripts directory to PATH (on macOS
typically `export PATH="$HOME/Library/Python/3.9/bin:$PATH"`) or call it as
`python3 -m geolm.cli`.

## Level 1. Self-check (one command, about a minute)

```bash
python3 tests/test_smoke.py
```

Expected output ends with a line like:

```
OK: training works, accuracy 7X.X%, save/load/inference intact
```

The test trains a tiny model to reverse strings, saves it, loads it back, and
compares answers. Passing proves the whole cycle works: training, saving, loading,
and prediction.

## Level 2. Full example: a calculator (a few minutes)

```bash
python3 examples/calculator.py
```

Expected: training progress lines with rising accuracy, then correct answers such as
`357+468=825`, and the message that `calc.geolm` was saved. Proves the library can
train a real working model from scratch.

## Level 3. Command line

```bash
geolm predict --model calc.geolm --input "357+468="        # expected: 825
printf "1+1=\n40+50=\n" | geolm predict --model calc.geolm  # expected: 2 and 90
geolm eval --model calc.geolm --data examples/data.jsonl    # expected: high % on 50 examples
```

Proves single, batch, and measured-accuracy modes all work.

## Level 4. Network service

Terminal one:

```bash
geolm serve --model calc.geolm --port 8080
```

Terminal two:

```bash
curl -X POST localhost:8080/predict -d '{"input": "357+468="}'
curl -X POST localhost:8080/predict -d '{"inputs": ["1+1=", "40+50="]}'
```

Expected: `{"output": "825"}` and `{"outputs": ["2", "90"]}`. Proves any application
in any language can call the model over HTTP. Stop the service with Ctrl+C.

## Level 5. Quality on your own data

Prepare `train.jsonl` and `holdout.jsonl` (JSONL, `input` and `output` fields).
Held-out examples must not appear in training data, otherwise the score is inflated.

```bash
geolm train --data train.jsonl --out my_task.geolm --steps 20000
geolm eval  --model my_task.geolm --data holdout.jsonl
```

The held-out exact-match percentage is the main quality number. Tuning rules: if
accuracy was still rising at the end, increase `--steps`; if training accuracy is
high but held-out accuracy is low, add more varied data; for harder tasks increase
the model (`--dim 256 --hidden 1024`).

## Level 6. Speed (optional)

```bash
python3 - <<'EOF'
import time
from geolm import GeoLM
model = GeoLM.load("calc.geolm")
qs = [f"{i%900}+{(i*7)%900}=" for i in range(1000)]
t0 = time.time()
model.predict_batch(qs)
dt = time.time() - t0
print(f"1000 answers in {dt:.2f}s, that is {1000/dt:.0f} queries per second")
EOF
```

Expected: hundreds to thousands of queries per second on a laptop CPU, tens of
thousands on a GPU.

## Common problems

The `geolm` command is not found: the scripts directory is not on PATH; use
`python3 -m geolm.cli` or extend PATH. Install fails on `pip install -e .`: upgrade
pip (`python3 -m pip install --user --upgrade pip`). Accuracy near zero: too few
steps, or too little varied data, or evaluation contains characters absent from
training data. Answers are truncated: retrain with a larger `--max-new`.

## Final checklist

The system is fully verified when six things hold: the package imports, the
self-check prints OK, the calculator example trains and answers correctly, predict
and eval work, the HTTP service answers POST requests, and you have a measured
held-out accuracy on your own data.
