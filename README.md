# geolm: narrow geometric models. Train for one task, ship to inference.

`geolm` trains a tiny neural network for one specific input-to-output task
(calculators, string normalization, classification, format conversion) and gives you
ready-to-use inference out of the box: a Python library, a command-line interface,
and an HTTP service. A trained model is a single file of a few megabytes.

Documentation in Russian: [README.ru.md](README.ru.md).

## Why not just a small transformer?

Inside `geolm` there is no attention. Every token is a learnable point in a
low-dimensional space, the sequence is read by a recurrent engine as a trajectory
through those points, and the next token is chosen by proximity to the predicted
point. One shared point cloud serves both input and output, so there is no separate
output vocabulary matrix, the cost of a long input grows linearly, and generation
memory is constant.

In a paired benchmark on the same task, with the same data and the same training
budget, at the same size (about 2.6M parameters, details in
[BENCHMARKS.md](BENCHMARKS.md)):

| Metric | geolm | transformer |
|---|---|---|
| Quality (answers within tolerance) | 91.5% | 91.6% |
| Model file | 10.0 MB | 10.2 MB |
| Latency per query | **0.03 ms** | 0.10 ms |
| Throughput (queries per second, one GPU) | **38,500** | 10,300 |
| Single-token latency at 32k context | **0.6 ms** | 23.2 ms |

The honest summary: same quality at the same size, roughly four times the serving
throughput, and flat latency at long context. The transformer still wins where broad
knowledge, long training runs at scale, or complex reasoning are required. The niche
of `geolm` is narrow high-load services, streaming pipelines, and weak hardware.

## Install

```bash
cd geolm
pip install -e .
```

The only dependency is PyTorch. Runs on CPU; uses CUDA automatically when available.

## Quickstart in 3 steps

Step 1. Prepare data as JSONL, one example per line, with `input` and `output` fields:

```json
{"input": "357+468=", "output": "825"}
{"input": "999-1=",   "output": "998"}
```

Step 2. Train with one command:

```bash
geolm train --data examples/data.jsonl --out task.geolm --steps 20000
```

Step 3. Use it:

```bash
geolm predict --model task.geolm --input "357+468="     # single answer
geolm eval    --model task.geolm --data holdout.jsonl   # exact-match accuracy
geolm serve   --model task.geolm --port 8080            # HTTP service
curl -X POST localhost:8080/predict -d '{"input": "357+468="}'
```

## Python API (scikit-learn style)

```python
from geolm import GeoLM

pairs = [("357+468=", "825"), ("999-1=", "998"), ...]   # (input, output)

model = GeoLM.new(pairs, dim=128, hidden=512)   # vocabulary is built from the data
model.fit(pairs, steps=20000)                   # trains, keeps the best checkpoint
model.save("task.geolm")                        # one file: weights + config + vocab

model = GeoLM.load("task.geolm")
model.predict("357+468=")                       # "825"
model.predict_batch(["1+1=", "2+2="])           # batched, fast
model.evaluate(holdout_pairs)                   # % of exact matches
```

A complete working example lives in `examples/calculator.py`: it generates data,
trains a calculator, prints answers, and saves `calc.geolm`.

## Choosing a model size

| Task | dim | hidden | layers | steps |
|---|---|---|---|---|
| Simple string transforms | 48-96 | 160-384 | 1-2 | 3-10k |
| Arithmetic, formats, extraction | 128 | 512 | 2 | 20-40k |
| Complex multi-step answers | 256 | 1024 | 2-4 | 40k+ |

Rule of thumb: if held-out accuracy is still climbing at the end of training, add
steps; if training accuracy is high but held-out accuracy is not, add data.

## Handling different kinds of input

The model consumes a string and produces a string. Structured data is serialized
into an agreed string format, and one model can serve several input types at once
if you prefix the task name in the training data (multi-task training):

```json
{"input": "date: 7 марта 1961", "output": "1961-03-07"}
{"input": "sum: 12+34",         "output": "46"}
```

Details with client examples for Python, JavaScript, Go, curl, and a Dockerfile:
[INTEGRATION.md](INTEGRATION.md).

## Testing

```bash
python tests/test_smoke.py    # about a minute on CPU
```

A full step-by-step testing guide, from install to quality and speed measurement on
your own data: [TESTING.md](TESTING.md).

## Honest limitations

- This is a specialist for one task, not a general assistant. For broad knowledge
  and complex reasoning use large transformers.
- Exact long division is hard (equally hard for a same-size transformer).
- The character vocabulary is built from training data: characters never seen in
  training are unknown to the model.
- All results were measured at small scale (millions of parameters). No claims are
  made about behavior at billions of parameters.

## Project layout

```
geolm/
├── pyproject.toml            # install with: pip install -e .
├── src/geolm/
│   ├── model.py              # geometric core: points + GRU + nearest-point output
│   ├── data.py               # vocabulary, JSONL, batch building with loss masking
│   ├── pipeline.py           # GeoLM: new / fit / save / load / predict / evaluate
│   └── cli.py                # geolm train | predict | eval | serve
├── examples/calculator.py    # end-to-end scenario from data to model
├── tests/test_smoke.py       # quick health check
├── BENCHMARKS.md             # full measured comparison against a transformer
├── INTEGRATION.md            # embedding into services (HTTP, Python, pipelines)
└── TESTING.md                # step-by-step testing guide
```

## Background

The architecture grew out of a series of experiments: representing tokens as points
in a low-dimensional space, treating a sentence as a trajectory through those
points, and predicting the next token by proximity to the predicted next position.
Related known ideas include tied input/output embeddings (Press and Wolf, 2017),
continuous-output language models (Kumar and Tsvetkov, 2019), and recurrent language
models; the combination of a very low-dimensional shared point cloud with a
recurrent trajectory engine, packaged as a practical tool for narrow tasks, is what
this project explores. Measurements are in [BENCHMARKS.md](BENCHMARKS.md).

## License

MIT.
