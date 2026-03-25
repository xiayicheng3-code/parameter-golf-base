# All-In Fixed Sliding M07

Cheap-screen experiment for the requested all-in stack:

`M02, M03, M05, M07, M09, M13, M15, M16, M18, M19, M20, M23, M25, M27, M28, M29, M30, M31`

## What changed

- Forks `2026-03-23_delta_init_export_hybrid` as the export / quantization base.
- Replaces full-attention eval tricks with native sliding-window attention in the actual model path.
- Keeps shifted attention as a fixed shallow-layer behavior on top of sliding-window attention.
- Replaces `resid_mix + skip_weights` with a unified `M07` hyper-connection module that sums the root state and all prior layer outputs before each pre-norm block.
- Swaps the FFN to SwiGLU while keeping the width decision compression-friendly.
- Adds the looping-layer trunk with per-pass LoRA adapters and MLP-side LayerRoPE.
- Uses step-wise loop-path operator fusion: each virtual pass materializes effective attention / MLP weights from base weights, LayerRoPE, and LoRA before the main matrix multiply.
- Removes residual carry-through inside the block: each layer stores only its newly produced output rather than adding the input back in.
- Keeps delta-hybrid export, higher-bit mixed compression, fp16-sensitive tensor protection, VE, BigramHash, SmearGate, Partial RoPE, q_gain, training-time weight noise, and deep-layer XSA.
- Supports multiple loop groups in the middle trunk, with optional non-loop bridge layers between groups.

## Fixed design choices

- Native sliding-window attention is the default and intended path for both train and validation.
- Shifted attention offsets are fixed in code: `1,2,3,4`.
- No `ATTENTION_IMPL` env knob.
- No `SHIFTED_ATTENTION_OFFSETS` env knob.
- No env knob for the unified `M07`.

## Useful knobs

```bash
SLIDING_WINDOW_SIZE=512
SLIDING_CHUNK_SIZE=0
SHIFTED_ATTENTION_LAYERS=3
NUM_PRELUDE_LAYERS=2
NUM_LOOP_GROUPS=1
NUM_LOOP_LAYERS=3
LOOP_REPEATS=2
NUM_INTER_LOOP_LAYERS=0
NUM_EPILOGUE_LAYERS=3
LORA_RANK=8
MLP_MULT=4.0
COMPRESSION_SCHEMES=delta_attn_int10_mlp_int8,delta_attn_int9_mlp_int7,delta_attn_int8_mlp_int6,raw_sota_int6_lzma,raw_int_mixed,delta_attn_int9_mlp_int8_sizeonly,delta_attn_int8_mlp_int7_sizeonly,delta_attn_int9_mlp_int7_sizeonly,delta_attn_int10_mlp_int7_sizeonly
EMA_ENABLED=0
WEIGHT_NOISE_ENABLED=1
WEIGHT_NOISE_SCALE=0.02
WEIGHT_NOISE_START_FRAC=0.3
```

The default depth layout is `2 + 3 x 2 + 3 = 11` effective layers, matching the default `NUM_LAYERS=11`.

## Intended readout

This branch is for viability screening, not score claims. The first question is:

Can this exact all-in stack train stably, preserve enough throughput, and still produce a plausible under-budget artifact after delta-hybrid export?

Primary failure modes to watch:

- throughput collapse from looping + sliding attention
- instability from dense M07 routing
- quantization/export regression from combining delta-hybrid with the new FFN and loop trunk

## Compression notes

- `raw_sota_int6_lzma` is the non-delta baseline aligned with the current record holder's storage style:
  per-row searched int6 for attention/MLP, int8 elsewhere, and `lzma` as the final blob compressor.
- The old `raw_gptq` label was removed because it was not a true GPTQ implementation.
- `int7` and `int9` are now stored with native packed bitstreams, not by leaving values in `int8/int16` shells.
- Schemes ending in `_sizeonly` still quantize and write the compressed artifact, but they skip roundtrip validation so you can cheaply compare packing size.
