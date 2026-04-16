# SP8192 LegalTTT M37 M38 Doc-Local Port v1

This experiment keeps the April 9, 2026 legal `SP8192` SOTA backbone intact and isolates only the data-side `M37 + M38` changes:

- `M37`: document-local BOS-aware train/eval windows
- `M38`: deterministic two-phase long-document coverage

Everything grammar-related from the previous graft is intentionally removed here. The goal is to answer the narrower question first: does the stronger document prior help on top of the current SOTA stack before we spend more time fixing the grammar path?

## Base

- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

`record_base.py` is copied locally so the run stays self-contained. `train_gpt.py` only patches the loader/eval path.

## What Changed

- Train windows never cross a document boundary.
- When the next real BOS exists, it is kept as the final prediction target for that document.
- Long documents alternate between left-packed and right-packed non-overlapping windows across passes.
- Non-BOS training windows mask their first `64` target positions by default so the model is not penalized for predicting from an artificially context-truncated opening.
- Eval uses document-local rolling coverage with overlap masking, so each target token is scored exactly once.
- Document-local TTT is still disabled here.

## Defaults

- `DOC_LOCAL_WINDOWS=1`
- `SHUFFLE_DOCS=1`
- `TRAIN_CONTEXT_BURNIN=64`
- `TTT_ENABLED=0`

Optional smoke knobs:

- `TRAIN_DOC_LIMIT`
- `VAL_DOC_LIMIT`
- `TRAIN_TOKEN_LIMIT`
- `VAL_TOKEN_LIMIT`

## Suggested First Run

```bash
RUN_ID=sp8192_m37_m38_smoke \
TTT_ENABLED=0 \
DOC_LOCAL_WINDOWS=1 \
TRAIN_CONTEXT_BURNIN=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py`: minimal M37/M38 patch layer
- `record_base.py`: copied April 9 legal SOTA base
- `submission.json`: experiment metadata
