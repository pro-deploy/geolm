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
