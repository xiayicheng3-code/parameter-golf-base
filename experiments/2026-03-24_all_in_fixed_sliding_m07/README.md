# Unified Record-Aligned Superset

Single-folder experiment script that now supports two runtime personalities in the same
[train_gpt.py](./train_gpt.py):

- `CONTROL_BASELINE=1` (default): record-style conservative control run
- `EXPERIMENTAL_ALL_IN=1`: re-enable the more aggressive backbone ideas when needed

The file intentionally keeps both paths, but control mode has explicit precedence so stale
experimental env vars do not silently reactivate the risky path.

## Default Control Profile

With no extra env vars, the script behaves like a record-aligned baseline:

- `CONTROL_BASELINE=1`
- `EXPERIMENTAL_ALL_IN=0`
- standard residual/U-Net style path, not full-history `M07`
- loop adapters off
- loop `LayerRoPE` off
- training-time weight noise off
- `EMA_ENABLED=1`
- `SWA_ENABLED=1`
- compile-safe late QAT enabled via `LATE_QAT_THRESHOLD=0.15`
- `MLP_ACTIVATION=leakyrelu2` with `LEAKY_RELU_SLOPE=0.5`
- `MLP_MULT=3.0`
- `BigramHash` on
- `SmearGate` on
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `VE_ENABLED=1`, `VE_DIM=128`
- `MUON_WD=0.04`, `ADAM_WD=0.04`
- `WARMDOWN_ITERS=3500`
- `GRAD_ACCUM_STEPS=0` uses the old automatic rule, but you can now set `GRAD_ACCUM_STEPS=1` for a true single-step large batch run

The default depth layout is also normalized into a plain 11-layer stack:

- `NUM_PRELUDE_LAYERS=0`
- `NUM_LOOP_GROUPS=1`
- `NUM_LOOP_LAYERS=NUM_LAYERS`
- `LOOP_REPEATS=1`
- `NUM_INTER_LOOP_LAYERS=0`
- `NUM_EPILOGUE_LAYERS=0`

This keeps the single file but makes the default run much closer to the record recipe.

## Experimental Path

The all-in machinery is still present for later reactivation:

- unified `M07` full-history hyper-connection
- loop groups
- per-pass loop LoRA adapters
- loop-path `LayerRoPE`
- native sliding/shifted attention as the main train path
- training-time weight noise

Recommended way to re-enable it:

```bash
CONTROL_BASELINE=0
EXPERIMENTAL_ALL_IN=1
USE_M07=1
USE_LOOP_ADAPTERS=1
USE_LAYER_ROPE=1
USE_SLIDING_ATTENTION_TRAIN=1
WEIGHT_NOISE_ENABLED=1
```

## Feature Precedence

Control mode wins over experimental toggles. When `CONTROL_BASELINE=1`, the script forcibly
disables:

- `USE_M07`
- `USE_LOOP_ADAPTERS`
- `USE_LAYER_ROPE`
- `USE_SLIDING_ATTENTION_TRAIN`
- shifted-attention train layers
- training-time weight noise

It also forces:

- `EMA_ENABLED=1`
- `SWA_ENABLED=1`
- `MLP_ACTIVATION=leakyrelu2`
- `LORA_RANK=0`

This is deliberate. The goal is to keep one superset script without letting env drift create
ambiguous hybrids.

## Quantization / Export

Main full-eval schemes kept in the same file:

- `raw_sota_int6_lzma`
- `raw_int_mixed`
- `delta_attn_int8_mlp_int6`
- `delta_attn_int10_mlp_int8`
- `delta_attn_int9_mlp_int7`
- `delta_attn_int7_mlp_int6`
- `delta_attn_int7_mlp_int5`
- `delta_attn_int6_mlp_int6`
- `delta_attn_int6_mlp_int5`
- `delta_attn_int6_mlp_int4`
- `delta_attn_int5_mlp_int4`

Size-probe only schemes:

- `delta_attn_int9_mlp_int8_sizeonly`
- `delta_attn_int8_mlp_int7_sizeonly`
- `delta_attn_int9_mlp_int7_sizeonly`
- `delta_attn_int10_mlp_int7_sizeonly`

Current default shortlist:

```bash
COMPRESSION_SCHEMES=raw_sota_int6_lzma,raw_int_mixed,delta_attn_int8_mlp_int6,delta_attn_int10_mlp_int8,delta_attn_int9_mlp_int7
```

Each scheme logs:

- source kind
- compressor
- payload bytes
- total submission bytes
- whether it ran full roundtrip eval or size-only

## TTT / Staged Features

Legal score-first TTT now lives in the same file and is optional:

```bash
TTT_ENABLED=1
TTT_LR=0.002
TTT_EPOCHS=3
EVAL_STRIDE=64
```

It is intended for control-baseline final evals, not for normal training diagnosis.

The following record-adjacent systems hooks are staged but intentionally not active yet:

- `PARAMETER_BANKING_ENABLED=1`
- `PARALLEL_MUON_ENABLED=1`

If either is set today, the script raises `NotImplementedError` rather than pretending to
support them.

## Suggested Control Run

```bash
RUN_ID=record_control \
CONTROL_BASELINE=1 \
EXPERIMENTAL_ALL_IN=0 \
EMA_ENABLED=1 \
SWA_ENABLED=1 \
WEIGHT_NOISE_ENABLED=0 \
COMPRESSION_SCHEMES=raw_sota_int6_lzma,raw_int_mixed,delta_attn_int8_mlp_int6,delta_attn_int10_mlp_int8,delta_attn_int9_mlp_int7 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

- `step 0` validation is disabled permanently.
- Late QAT is implemented with per-module tensor gates rather than a pure class-level flag so it
  remains active under `torch.compile`.
- `INT7`, `INT9`, and `INT10` keep native packed export support in this same file.
- `INT9` uses streaming bit packing to avoid the old 72-bit overflow bug from the `uint64` path.
