# Benchmarks: geolm versus a same-size transformer

All numbers below were actually measured, not estimated. Hardware: NVIDIA GeForce
RTX 4060 Ti 16 GB unless stated otherwise. Both models were always trained on the
same data, with the same context length, the same batch size, and the same number of
steps. The transformer baseline uses pre-norm blocks, multi-head attention with
flash attention kernels, a learning-rate warmup with cosine decay, dropout, and
weight decay, so it is a properly tuned baseline rather than a strawman.

An honest scope note: everything here is small scale (millions of parameters, up to
hundreds of millions of tokens). These results support conclusions about narrow
specialized models and serving efficiency. They make no claims about frontier-scale
behavior, where transformers have well-established advantages.

## 1. Paired task benchmark: engineering calculator

One shared task: ten operations (addition, subtraction, multiplication, division,
power, square root, sine, cosine, logarithm, exponent) in a character-level
sequence-to-sequence setup, answers formatted to four decimal places. Both models
about 2.6M parameters, 40,000 steps, batch 256.

Correct answers within a 0.01 tolerance, per operation, on 3,000 fresh examples per
operation:

| Operation | geolm | transformer |
|---|---|---|
| Addition | 99.5% | 99.9% |
| Subtraction | 99.5% | 100.0% |
| Multiplication | 99.9% | 100.0% |
| Division | 15.2% | 14.2% |
| Power | 100.0% | 100.0% |
| Square root | 99.9% | 100.0% |
| Sine | 100.0% | 100.0% |
| Cosine | 100.0% | 100.0% |
| Logarithm | 99.9% | 100.0% |
| Exponent | 100.0% | 100.0% |

Both models fail equally on exact many-digit division, which is a known hard case at
this size. Aggregates:

| Metric | geolm | transformer |
|---|---|---|
| Overall within tolerance | 91.5% | 91.6% |
| Model file (fp32) | 10.0 MB | 10.2 MB |
| Training throughput | 244k tokens/s | 172k tokens/s |
| Inference latency per query | 0.03 ms | 0.10 ms |
| Inference throughput | 38,521 queries/s | 10,349 queries/s |

Note on an earlier shorter run (8,000 steps): the transformer was ahead (68.2%
against 48.0% overall) because the geometric model converges more slowly on this
mixture; with an adequate budget the gap disappears. Both facts are reported for
honesty.

## 2. Context-length scaling at generation time

Architectural measurement (untrained weights, since latency and memory depend on
architecture, not on what the weights have learned). geolm 16.6M parameters,
transformer 27.8M with a comparable backbone. Latency is per generated token after
prefilling a prompt of the given length; peak GPU memory includes the prefill.

| Context | geolm memory | geolm ms/token | transformer memory | transformer ms/token |
|---|---|---|---|---|
| 512 | 0.21 GB | 0.6 | 0.19 GB | 4.6 |
| 1,024 | 0.24 GB | 0.6 | 0.20 GB | 4.3 |
| 2,048 | 0.30 GB | 0.6 | 0.23 GB | 4.3 |
| 4,096 | 0.41 GB | 0.6 | 0.29 GB | 4.5 |
| 8,192 | 0.63 GB | 0.5 | 0.41 GB | 6.0 |
| 16,384 | 1.08 GB | 0.6 | 0.64 GB | 11.7 |
| 32,768 | 1.97 GB | 0.6 | 1.11 GB | 23.2 |

Reading this honestly: per-token latency of the geometric model is flat (0.6 ms at
any length), while the transformer grows with context because each new token attends
to the whole growing key-value cache; at 32k context the difference is about forty
times. Peak memory in this table grows for both because it includes prefill
activations over the whole prompt; the state carried between generated tokens is
constant-size for geolm and grows linearly (key-value cache) for the transformer.
Training throughput at context 512 favored the transformer (55k against 25k
tokens/s) thanks to parallelism over the sequence; that is reported as-is.

## 3. Small-data language modeling

Russian Wikipedia corpus, 3.4M tokens, context 128, identical training budget with
warmup, cosine decay, dropout, and weight decay for all models. Metric: top-1
next-token accuracy on a held-out split.

| Model | Parameters | Top-1 accuracy |
|---|---|---|
| geolm-style (GRU + geometric output) | 1.84M | **14.5%** |
| Transformer, same size | 1.63M | 7.3% |
| Transformer, four times larger | 7.04M | 9.6% |

At this data scale the geometric model was markedly more data-efficient and did not
overfit; the same-size transformer needed careful tuning and still trailed. This
advantage is expected to shrink and reverse as data grows.

## 4. Engine variants and ternary quantization

Python code corpus (150M tokens of permissively collected public code), 4M-parameter
models, identical budgets, top-1 next-token accuracy:

| Variant | Top-1 |
|---|---|
| Geometric output + GRU engine | 57.6% |
| Geometric output + Mamba-style selective SSM engine | **59.0%** |

Ternary quantization of the point cloud (3 levels per axis) on a small model:
post-training quantization destroys the model (59.0% drops to 0.0%), while
quantization-aware training recovers most of the quality (53.7% at 3 levels, 58.2%
at 9 levels). Conclusion: the geometric point cloud is ternary-compressible, but
only when trained on the grid from the start.

## Reproduction

The paired calculator benchmark and the error analysis scripts used for these tables
live in the research archive alongside this project (`calc_vs.py`, `analyze.py`,
`compare.py`). The library itself reproduces the calculator setup via
`examples/calculator.py` plus `geolm eval`.
