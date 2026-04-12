# SP8192 LegalTTT Grammar Port v1

This experiment starts from the published legal April 9, 2026 SP8192 record and ports over the web-grammar work as a clean graft rather than a fork full of logs.

## Base

The baseline architecture and training stack come from:

- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

Attribution for that record remains with the original authors listed in its README. This experiment keeps a plain-text copy of the decompressed record code in [`record_base.py`](./record_base.py), and layers the grammar/data-path changes in [`train_gpt.py`](./train_gpt.py).

## Ported Ideas

- Input-only grammar adapter: the 36 cross-token grammar state features are added to token embeddings and never act as a count-table or logit prior.
- Document-local training windows: training and eval recover document boundaries from BOS, keep windows inside one document, and use deterministic two-phase long-document coverage.
- CPU grammar workers: workers build full-document grammar states, cache moderate-size documents, ship `uint16` tokens and `uint8` feature ids, and hand off through explicit host/device prefetch.
- Optional early local blocks: the first `LOCAL_ATTN_LAYERS` can switch from pure FlashAttention-3 global attention to chunked sliding-window attention plus causal depthwise convolution.

## Defaults

The defaults are intentionally conservative:

- `DOC_LOCAL_WINDOWS=1`
- `USE_GRAMMAR=1`
- `GRAMMAR_INIT_SCALE=0.0`
- `LOCAL_ATTN_LAYERS=0`
- `LOCAL_CONV_KERNEL=0`

That means the document-local data prior and grammar adapter are on by default, but the early local attention/conv path stays off unless we explicitly ablate it. On Hopper this preserves the record model's FlashAttention-3 fast path for all layers by default; on non-Hopper CUDA it can now fall back to FlashAttention-2 and then SDPA.

## GPU Path Audit

This graft keeps or adds the main throughput protections we care about:

- Mixed precision stays on: training/eval still use `torch.autocast(..., dtype=torch.bfloat16)`.
- `torch.compile` stays on for the main model and quantized eval model.
- Global attention now dispatches by backend: FlashAttention-3 on Hopper, FlashAttention-2 on other CUDA GPUs when available, and PyTorch SDPA otherwise. Only the optional early local layers use the chunked sliding-window SDPA path.
- No gradient checkpointing was added; this remains a pure DDP path.
- Per-step `.item()` calls are still restricted to logging checkpoints, not every micro-step.
- The dataloader path now uses document workers, worker feature caching, pinned host staging, and optional host-side prefetch overlap.
- `ATTN_BACKEND=auto` is the default; you can override with `fa3`, `fa2`, or `sdpa` when debugging platform-specific issues.

## Current Caveat

- Document-local TTT is not ported yet. Run this experiment with `TTT_ENABLED=0`.

## Suggested First Run

```bash
RUN_ID=sp8192_grammar_port_smoke \
TTT_ENABLED=0 \
DOC_LOCAL_WINDOWS=1 \
USE_GRAMMAR=1 \
LOCAL_ATTN_LAYERS=0 \
NUM_WORKERS=4 \
HOST_PREFETCH_BATCHES=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- [`train_gpt.py`](./train_gpt.py): patch layer for the record model
- [`record_base.py`](./record_base.py): decompressed April 9 legal SOTA base
- [`grammar_features.py`](./grammar_features.py): 36-state grammar feature builder with follow-up edge, member-chain, and key/value transition features
- [`submission.json`](./submission.json): experiment metadata
