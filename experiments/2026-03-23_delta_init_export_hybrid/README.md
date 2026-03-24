# Delta Init Export Hybrid

This experiment forks the 2026-03-22 11-layer SOTA stack and changes the
initialization/export story to better support `Delta = W - W0`.
One training run now emits multiple compressed artifacts and evaluates each
roundtrip separately.

## What changed

- Adds `INIT_SEED` and `INIT_IMPL=stateless_ortho_v1` so the reference weights `W0`
  can be regenerated without depending on incidental RNG call order.
- Sets `LATE_QAT_THRESHOLD=0` by default, because this branch is focused on
  throughput-first delta export rather than late fake-quant training.
- Adds `EXPORT_MODE=delta_hybrid`:
  - `attn.c_q`, `attn.c_k`, `attn.c_v`, and `mlp.fc` are exported as deltas from `W0`
  - zero-init-style tensors such as `attn.proj`, `mlp.proj`, `bigram`, and `ve.proj`
    stay on the raw path
- Adds `DELTA_WEIGHT_DECAY=1` as an optional centered decay mode that shrinks
  parameters toward `W0` instead of toward zero.
- Replaces the single export path with a compression sweep. The default schemes are:
  - `delta_attn_fp8_mlp_fp6`
  - `delta_attn_int8_mlp_int6`
  - `delta_attn_fp6_mlp_fp6`
  - `raw_gptq`
  - `raw_int_mixed`
- Disables sliding-window eval by default (`EVAL_STRIDE=0`) so smoke tests and
  compression sweeps reflect the normal eval path first.

## Default intent

The default configuration is designed to answer a narrow question first:

Can the current SOTA gain artifact headroom from hybrid delta export without
relying on QAT, and which post-training quantization family behaves best?

The expected tradeoff is:

- better compression on orthogonal-init matrices
- little or no gain on zero-init output projections
- slightly different quantization behavior because export now mixes raw and delta
  tensors

## Useful knobs

```bash
INIT_SEED=1337
INIT_IMPL=stateless_ortho_v1
EXPORT_MODE=delta_hybrid
COMPRESSION_SCHEMES=delta_attn_fp8_mlp_fp6,delta_attn_int8_mlp_int6,delta_attn_fp6_mlp_fp6,raw_gptq,raw_int_mixed
LATE_QAT_THRESHOLD=0
DELTA_WEIGHT_DECAY=0
DELTA_FP16_NAME_PATTERNS=
```

## Scheme meanings

- `delta_attn_fp8_mlp_fp6`: hybrid delta source, minifloat FP8 for attention-like
  tensors and FP6 for MLP-like tensors
- `delta_attn_int8_mlp_int6`: hybrid delta source, integer quantization baseline
  inside the delta family
- `delta_attn_fp6_mlp_fp6`: most aggressive all-minifloat delta variant
- `raw_gptq`: closest reference to the current 03-22 exporter, using GPTQ-lite
  int6 on attention/MLP and int8 elsewhere
- `raw_int_mixed`: non-GPTQ reference, using fixed-clip int6/int8 mixed export

## Suggested next comparison

Run matched seeds with:

1. `DELTA_WEIGHT_DECAY=0`
2. `DELTA_WEIGHT_DECAY=1`

The per-scheme logs then tell us whether centered decay helps the delta family
and whether any minifloat combination beats the integer references on the
size/quality frontier.
