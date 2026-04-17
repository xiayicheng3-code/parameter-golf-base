# SP8192 LegalTTT M37 M38 Doc-Local Port v1

This experiment keeps the April 9, 2026 legal `SP8192` SOTA backbone intact and isolates only the data-side `M37 + M38` changes:

- `M37`: document-local BOS-aware train/eval windows
- `M38`: deterministic two-phase long-document coverage

Everything grammar-related from the previous graft is intentionally removed here. The goal is to answer the narrower question first: does the stronger document prior help on top of the current SOTA stack before we spend more time fixing the grammar path?

## Base

- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

`train_gpt.py` is now a single self-contained script that folds the April 9 legal SP8192 SOTA base together with the M37/M38 doc-local patch layer.

## What Changed

- Train windows never cross a document boundary.
- When the next real BOS exists, it is kept as the final prediction target for that document.
- Long documents alternate between left-packed and right-packed non-overlapping windows across passes.
- Non-BOS training windows mask their first `64` target positions by default so the model is not penalized for predicting from an artificially context-truncated opening.
- Eval uses document-local rolling coverage with overlap masking, so each target token is scored exactly once.
- Optional document-local TTT now groups full documents into variable-size eval chunks, allows oversize chunks for very long documents, and drops any final remainder below one full global train batch.

## Defaults

- Document-local windows are always enabled in this script.
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
TRAIN_CONTEXT_BURNIN=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

TTT run:

```bash
RUN_ID=sp8192_m37_m38_ttt \
TTT_ENABLED=1 \
TRAIN_CONTEXT_BURNIN=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py`: self-contained April 9 legal SP8192 base plus the M37/M38 doc-local graft
- `submission.json`: experiment metadata
