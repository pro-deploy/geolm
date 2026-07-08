# Changelog

## 0.1.0 (2026-07-08)

First public release.

- Geometric core: learnable token point cloud, GRU trajectory engine, nearest-point
  output with a shared input/output cloud.
- High-level API: GeoLM.new / fit / save / load / predict / predict_batch / evaluate.
- Command line: geolm train, predict, eval, serve (HTTP service on the standard library).
- Single-file model format (weights + config + vocabulary).
- Correct batched inference for variable-length inputs via packed sequences.
- Docs: README, BENCHMARKS (measured comparison against a same-size transformer),
  TESTING and INTEGRATION guides in English and Russian.
