# Results: residual depth and scratchpad reasoning (division task)

2,000 held-out examples, all variants about 2.6M parameters, 12,000 steps each,
same data and budget. Hardware: RTX 4060 Ti.

| Variant | Exact answer | Within 0.01 | Avg output chars | ms per query |
|---|---|---|---|---|
| base + direct | 85.0% | 85.1% | 6.5 | 0.82 |
| deep + direct (residual blocks) | **99.4%** | 99.4% | 6.5 | 0.86 |
| base + reasoning (scratchpad) | 90.6% | 90.8% | 30.3 | 3.25 |
| deep + reasoning | **99.6%** | 99.6% | 30.3 | 2.66 |

Findings. Transformer-style residual stacking is the efficient win: at the same
parameter count and the same latency it lifted division from 85.0% to 99.4%.
Scratchpad reasoning helps the shallow model (+5.6 points) but costs about four
times the output length and latency; on top of the deep model it adds almost
nothing here. Practical rule: add depth first, reserve reasoning for tasks where
depth alone is not enough. Note that a dedicated division specialist reaches 85%
even in the base architecture, against 15% for the same architecture inside the
mixed ten-operation calculator: narrow specialists benefit from a dedicated budget.

# Results: per-layer token embeddings (PLE) on the mixed calculator

5,000 held-out examples, ten operations, all variants about 2.6M parameters,
20,000 steps each (half the budget of the original calculator benchmark, hence
lower absolute numbers; the comparison inside the table is fair).

| Operation | base | deep (residual) | deep + PLE |
|---|---|---|---|
| add | 63.3% | **85.1%** | 81.4% |
| sub | 59.7% | **80.4%** | 80.3% |
| mul | 84.3% | **100.0%** | 100.0% |
| div | 25.9% | 29.2% | 28.5% |
| pow, sqrt, sin, cos, log, exp | 100% | 100% | 100% |
| ALL | 83.8% | **89.6%** | 89.3% |

Findings. Residual depth wins again: plus 5.8 points overall at the same size,
with the largest gains exactly on exact multi-digit arithmetic (addition 63 to 85,
subtraction 60 to 80, multiplication 84 to 100). Per-layer token embeddings add
nothing on top (89.3% against 89.6%): with a tiny character-level vocabulary the
capacity benefit of PLE vanishes, and pure token injection is redundant next to
the residual stream. PLE remains a candidate for large-vocabulary variants only.
Division stays the bottleneck in the mixed setting at this budget, consistent
with the earlier finding that it needs either a dedicated specialist budget or a
scratchpad.
