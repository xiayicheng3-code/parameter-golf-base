`M51` is a minimal baseline fork of the April 9 legal SP8192 record.

Change:

- Replace the fixed `F.leaky_relu(..., negative_slope=0.5).square()` FFN activation with a layer-dependent slope:
  - `negative_slope = LAYER_LEAKY_BASE + LAYER_LEAKY_STEP * layer_idx`
  - default `LAYER_LEAKY_BASE=0.25`
  - default `LAYER_LEAKY_STEP=0.05`
  - for physical layers `L=0..10`, this gives slopes `0.25, 0.30, ..., 0.75`

Intent:

- Keep throughput essentially unchanged while testing whether the observed shallow-vs-deep activation preference can be applied with almost no extra machinery.
