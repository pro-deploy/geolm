# Geometric Language Models: Experimental Study

This document reports, in the format of a paper's experimental section, the full
set of experiments behind `geolm`. It describes the model, the common setup, and
each experiment with its motivation, method, quantitative results, and findings.
All numbers were measured, not estimated. Negative results are reported as such.

## 1. Model

We study an autoregressive sequence model whose output is geometric rather than a
softmax over a large projection matrix. Every token is a learnable point in a
low-dimensional space (the point cloud). A sequence is read by a recurrent or
deep-residual engine that predicts, at each step, the coordinates of the next
point. The next-token distribution is defined by proximity of the predicted point
to the points of the shared cloud, using the negative squared Euclidean distance
scaled by a learnable temperature:

    logits(f)_v = -||f - E_v||^2 * exp(s),

where f is the predicted point, E_v is the cloud point of token v, and s is a
scalar. The same point cloud E is used for input embedding and for output scoring
(fully tied). Consequences of this design: there is no separate output vocabulary
matrix; the representation is low-dimensional and interpretable; with a recurrent
engine the generation state is constant size and per-token cost is independent of
context length.

Two engines are used. (a) A stacked GRU (the default). (b) A deep residual stack,
where each block applies pre-normalization, a sequence mixer, and a residual add,
so intermediate values flow past every layer untouched (the transformer lesson
applied to a recurrent mixer).

## 2. Common setup

Unless stated otherwise, models are trained with AdamW, a linear warmup followed
by cosine decay, gradient clipping at 1.0, and dropout 0.1. Quality on formal
tasks is measured as exact-match accuracy on held-out data; for approximate
operations we also report accuracy within a 0.01 tolerance. For language modeling
we report top-1 next-token accuracy or bits per token/byte. Hardware: a single
NVIDIA RTX 4060 Ti (16 GB) or RTX 5060 Ti (16 GB) unless noted. All comparisons
against a transformer use a pre-norm baseline with multi-head attention and flash
attention kernels, warmup, dropout, and weight decay, i.e. a properly tuned
baseline, not a strawman.

Scope note. All experiments are small scale (millions of parameters, up to
hundreds of millions of tokens). Conclusions concern narrow specialized models,
serving efficiency, and architectural ablations. No claim is made about
frontier-scale behavior, where transformers have established advantages.

## 3. Motivating studies: why a learned geometry is necessary

### 3.1 Static geometric coordinates carry no predictive signal by themselves

We first asked whether tokens placed at fixed low-dimensional coordinates, without
learning, could support prediction. Encoding each token of a 50k vocabulary as a
unique ternary 3D coordinate compresses the identifier to about 2 bytes per token
(roughly 20 to 29 times smaller than the raw vocabulary strings), but such a
coordinate is only an index and carries no meaning. Using pretrained token
embeddings projected to three dimensions retains only 3.4% of the embedding
variance; quantizing that 3D space into cells and predicting the next token from a
coarsened context yields at most about 7.7% top-1 on Russian text, and increasing
the number of ternary dimensions is non-monotonic: accuracy peaks at two to four
dimensions and then reverts to memorization as cells become unique. Clustering the
full 768-dimensional embeddings into a fixed number of classes did not beat the
crude 3D scheme (about 7.5% top-1), showing that the ceiling was set by the
count-based prediction mechanism, not by the coordinate scheme.

### 3.2 Route memorization does not generalize

Modeling language as stored routes between token points (an n-gram transition
graph) reaches at most about 14% top-1 with a one-token context on Russian text
and degrades with longer context as coverage collapses; the model can only
reproduce sequences seen verbatim. This established the need for a learned model
that generalizes rather than memorizes routes, which motivates the trainable
geometric model studied below.

## 4. Data efficiency at small scale

Corpus: Russian Wikipedia, 3.4M tokens, context 128, identical training budget for
all models. Metric: top-1 next-token accuracy on a held-out split.

| Model | Parameters | Top-1 accuracy |
|---|---|---|
| Geometric (GRU + nearest-point output) | 1.84M | **14.5%** |
| Transformer, same size | 1.63M | 7.3% |
| Transformer, four times larger | 7.04M | 9.6% |

Finding. At this data scale the geometric model is markedly more data-efficient
and does not overfit, while the same-size transformer needs careful tuning and
still trails, even at four times the parameters. This advantage is expected to
shrink and reverse as data grows.

## 5. Code language modeling

Corpus: permissively collected public Python code, GPT-2 byte-level BPE tokenizer,
150M tokens, context 256. A 23M-parameter GRU geometric model reached 69% top-1
next-token accuracy on a held-out split. On the independent HumanEval benchmark
(164 problems, code never seen in training) it reached 55.5% top-1 next-token
accuracy, confirming generalization to out-of-distribution code. Free-form
generations were syntactically valid Python with correct control flow, class
definitions, and idiomatic NumPy-style docstrings, though not semantically correct
at this scale. This demonstrates that the geometric output supports real language
modeling, not only formal toy tasks.

## 6. Engine ablation: recurrent versus selective state space

Corpus: Python code, 4M-parameter models, identical budget, top-1 next-token
accuracy. The geometric output is held fixed; only the sequence engine changes.

| Engine (with geometric output) | Top-1 |
|---|---|
| GRU | 57.6% |
| Selective state-space (Mamba-style) | **59.0%** |

Finding. The geometric output is engine-agnostic and benefits from a modern
selective state-space mixer, which was also more sample-efficient during training.
Both engines are linear in context length.

## 7. Ternary quantization of the point cloud

We quantized the point-cloud coordinates to a ternary grid (three levels per
axis). Post-training quantization of a trained continuous model destroys it;
quantization-aware training (straight-through estimator on the grid from the start)
recovers most of the quality.

| Regime | Top-1 |
|---|---|
| Continuous (reference) | 59.0% |
| Post-training ternary quantization | 0.0% |
| Quantization-aware training, 3 levels | 53.7% |
| Quantization-aware training, 9 levels | 58.2% |

Finding. The geometric point cloud is ternary-compressible, but only when trained
on the grid from the beginning; the information a well-trained model stores lives
in continuous coordinate precision and cannot be recovered by naive post-hoc
snapping.

## 8. Paired benchmark: geometric output versus transformer

Task: a mixed ten-operation engineering calculator (addition, subtraction,
multiplication, division, power, square root, sine, cosine, logarithm, exponent),
character-level sequence-to-sequence, answers to four decimal places. Both models
about 2.6M parameters, same data and budget, trained to convergence (40,000 steps).
Answers within a 0.01 tolerance.

| Operation | Geometric | Transformer |
|---|---|---|
| Addition | 99.5% | 99.9% |
| Subtraction | 99.5% | 100.0% |
| Multiplication | 99.9% | 100.0% |
| Division | 15.2% | 14.2% |
| Power, sqrt, sin, cos, log, exp | ~100% | ~100% |
| Overall | 91.5% | 91.6% |

Per-operation error analysis on 3,000 fresh examples per operation (exact match
for exact arithmetic, tolerance for functions): both models make under 2% errors
on every operation except division, where both fail about 84% of the time.
Division to four exact decimals is a shared hard case at this size, not an
architecture difference; excluding it, both models are about 99% correct.

Efficiency (same 2.6M size, same hardware):

| Metric | Geometric | Transformer |
|---|---|---|
| Model file (fp32) | 10.0 MB | 10.2 MB |
| Training throughput (short sequences) | 244k tok/s | 172k tok/s |
| Inference latency per query | 0.03 ms | 0.10 ms |
| Inference throughput | 38,521 q/s | 10,349 q/s |

Finding. At equal size and training the geometric model matches transformer
accuracy operation by operation, while serving about 3.7 times more queries per
second at one third the latency, and training faster on short sequences. An earlier
under-trained run (8,000 steps) showed the transformer ahead (68.2% against 48.0%),
because the geometric model converges more slowly on this mixture; with an adequate
budget the gap closes. Both facts are reported.

## 9. Inference efficiency versus context length

Architectural measurement (randomly initialized matched models, since latency and
memory depend on architecture, not on learned weights). Geometric model 16.6M
parameters, transformer 27.8M with a comparable backbone and a flash-attention
key-value cache. Latency is per generated token after prefilling a prompt of the
given length; peak memory includes the prefill.

| Context | Geo mem | Geo ms/token | Transformer mem | Transformer ms/token |
|---|---|---|---|---|
| 512 | 0.21 GB | 0.6 | 0.19 GB | 4.6 |
| 1,024 | 0.24 GB | 0.6 | 0.20 GB | 4.3 |
| 2,048 | 0.30 GB | 0.6 | 0.23 GB | 4.3 |
| 4,096 | 0.41 GB | 0.6 | 0.29 GB | 4.5 |
| 8,192 | 0.63 GB | 0.5 | 0.41 GB | 6.0 |
| 16,384 | 1.08 GB | 0.6 | 0.64 GB | 11.7 |
| 32,768 | 1.97 GB | 0.6 | 1.11 GB | 23.2 |

Finding. Per-token latency of the geometric model is flat (about 0.6 ms at any
length), while the transformer grows because each generated token attends to the
whole key-value cache; at 32k context the difference is about forty times. The
transformer additionally has a fixed maximum context from its position embeddings,
which the recurrent geometric model does not. The recurrent generation state is
constant size; the transformer key-value cache grows linearly. Training throughput
at long context favors the transformer (55k against 25k tok/s at context 512) due
to parallelism over the sequence; this is reported as-is.

## 10. Architectural improvement: residual depth

We compared the default stacked engine against a deep residual stack (pre-norm
blocks with a residual stream). Task: division to four decimals (the failure case
from Section 8), 2,000 held-out examples, about 2.6M parameters, 12,000 steps.

| Variant | Exact answer | Avg output chars | ms/query |
|---|---|---|---|
| Base, direct answer | 85.0% | 6.5 | 0.82 |
| Deep residual, direct answer | **99.4%** | 6.5 | 0.86 |

The result was confirmed on the full mixed calculator (Section 12): residual depth
raised the overall score from 83.8% to 89.6% at the same size, with the largest
gains on exact multi-digit arithmetic (addition 63.3 to 85.1, subtraction 59.7 to
80.4, multiplication 84.3 to 100.0).

Finding. Transformer-style residual stacking is a near-free improvement for the
geometric model: at the same parameter count and the same latency it substantially
raises accuracy on hard multi-step operations.

## 11. Reasoning via an explicit scratchpad

We trained the model to emit an intermediate long-division scratchpad (successive
quotient digits and remainders) before the final answer, versus answering directly.
Division task, same setup as Section 10.

| Variant | Exact answer | Avg output chars | ms/query |
|---|---|---|---|
| Base, direct | 85.0% | 6.5 | 0.82 |
| Base, scratchpad | 90.6% | 30.3 | 3.25 |
| Deep, direct | 99.4% | 6.5 | 0.86 |
| Deep, scratchpad | 99.6% | 30.3 | 2.66 |

Finding. Scratchpad reasoning helps the shallow model (+5.6 points) by moving
intermediate remainders out of the compressed state and into the token stream, but
costs about four times the output length and inference latency; on top of an
already deep model it adds almost nothing here. Practical rule: add depth first,
reserve reasoning for tasks where depth alone is insufficient.

## 12. Per-layer embeddings: a negative result

Motivated by per-layer embeddings (PLE) in recent transformer families, we gave
each residual block its own embedding of the current input token, injected into
that block. Mixed calculator, 5,000 held-out examples, about 2.6M parameters,
20,000 steps.

| | Base | Deep residual | Deep + PLE |
|---|---|---|---|
| Overall within tolerance | 83.8% | **89.6%** | 89.3% |

Finding. PLE adds nothing on top of the residual stack in our setting. With a small
character-level vocabulary the capacity benefit of PLE (its main source of value at
large vocabularies) vanishes, and pure per-layer token injection is redundant next
to the residual stream. PLE remains a candidate only for large-vocabulary variants.

## 13. Natural-language pretraining (preliminary)

We trained a 34M-parameter deep residual geometric model on 300M tokens of
Russian technical articles (Habr), with a task-specific 32k byte-level BPE
tokenizer, following the recipe of a public habrGPT replication. The run was
interrupted at 8,000 of 20,000 steps by resource contention on a shared GPU
(about 2 bits per byte at that point). Generations reproduced the surface form of
technical articles (valid Russian grammar, section headings, technical vocabulary,
site-specific phrasing) but lacked global coherence, and code generation was poor,
as expected for an under-trained sub-100M model on a text-heavy corpus with little
code. A larger 68M two-GPU run reached about 1.08 bits per byte after 1,000 steps
before being stopped. These natural-language results are preliminary and reported
only to document that the architecture produces connected text; full training on
uncontended hardware is future work.

## 14. Summary of findings

1. A learned geometric output on a recurrent or deep-residual engine is a viable
   autoregressive model: it matches a same-size transformer on formal tasks
   (Section 8) and does real code and text language modeling (Sections 5, 13).
2. Its measured advantages are serving efficiency: about 3.7 times the query
   throughput at equal size and accuracy (Section 8), flat per-token latency and
   no fixed context limit at long context (Section 9), a compact ternary-
   compressible representation (Section 7), and data efficiency at small scale
   (Section 4).
3. Its costs are slower training on long sequences and a slower convergence on
   mixed tasks; at scale, transformers retain advantages in broad knowledge and
   complex reasoning.
4. Two architectural transfers help: residual depth is a near-free gain
   (Sections 10, 12); explicit scratchpad reasoning helps shallow models at a
   latency cost (Section 11). Per-layer embeddings do not help at small vocabulary
   (Section 12).

## 15. Limitations

All results are at millions of parameters and up to hundreds of millions of tokens.
Exact many-digit division is a shared failure case with same-size transformers.
The character-level vocabulary cannot represent unseen symbols. The natural-language
pretraining is preliminary and interrupted. No frontier-scale claims are made.

## 16. Reproducibility

The library reproduces the calculator setup via `examples/calculator.py` and
`geolm eval`. The architectural ablations (residual depth, scratchpad reasoning,
per-layer embeddings) are in `experiments/deep_reasoning.py` and
`experiments/ple_geometry.py`. The transformer comparison and inference-scaling
harnesses are archived alongside the research code. Every table above was produced
by a script; seeds are fixed where applicable.
