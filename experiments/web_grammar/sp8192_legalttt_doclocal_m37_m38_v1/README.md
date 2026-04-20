# SP8192 LegalTTT M37 M38 Doc-Local Port v1

This experiment started as a narrow April 9, 2026 `SP8192` LegalTTT port focused on data-side `M37 + M38` changes:

- `M37`: document-local BOS-aware train/eval windows
- `M38`: deterministic two-phase long-document coverage

It has since evolved into a broader data-loader sandbox for `SP8192`, because the first doc-local version exposed a large short-document inefficiency under the `SP8192` tokenizer.

## Base

- `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

`train_gpt.py` is a single self-contained script that folds the April 9 legal SP8192 SOTA base together with the current data-loader experiments.

## What Changed

- Current default train/eval regime is `1024`, not `2048`.
- Short documents (`pred_len < 1024`) are packed together until the concatenated length first reaches `1024`; the prefix `1024` tokens are kept and overflow is dropped.
- Long documents (`pred_len >= 1024`) still use deterministic two-phase windowing.
- Non-BOS long-document train windows mask their first `32` target positions by default.
- Eval uses `1024` document-local rolling coverage with overlap masking, so each scored target token is counted once.
- Optional document-local TTT still runs on document chunks, but now under the `1024` eval regime.

## Defaults

- `TRAIN_SEQ_LEN=1024`
- `EVAL_SEQ_LEN=1024`
- `SHUFFLE_DOCS=1`
- `TRAIN_CONTEXT_BURNIN=32`
- `TTT_ENABLED=0`
- `GPTQ_RESERVE_SECONDS=15`

Optional smoke knobs:

- `TRAIN_DOC_LIMIT`
- `VAL_DOC_LIMIT`
- `TRAIN_TOKEN_LIMIT`
- `VAL_TOKEN_LIMIT`

## Suggested First Run

```bash
RUN_ID=sp8192_m37_m38_smoke \
TTT_ENABLED=0 \
TRAIN_CONTEXT_BURNIN=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

TTT run:

```bash
RUN_ID=sp8192_m37_m38_ttt \
TTT_ENABLED=1 \
TRAIN_CONTEXT_BURNIN=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Apr 18 Note

- The April 18 `1024 + short-doc packing` run should not be compared against the April 9 plain pre-quant metric directly. In this branch, the logged `pre-quantization post-ema` number was already a rolling/doc-local eval, not the original plain non-overlap pre-quant metric.
- Within the same rolling metric, GPTQ still caused a real drop:
  - pre-quant rolling: `1.07641879` bpb
  - quantized rolling: `1.08902988` bpb
  - quantized TTT: `1.08693537` bpb
- Compared with the April 9 baseline, the extra quantization penalty is only about `+0.0012` bpb. That explains roughly `20%` of the observed `~0.006` gap.
- The remaining `~0.005` gap is more likely caused by the training/data regime itself: short-doc packing details, `1024` sequence length, the small first-layer FFN width cut, and the changed eval regime.
- Practical takeaway: quantization is a secondary amplifier here, not the main reason this branch regressed.

## Files

- `train_gpt.py`: self-contained April 9 legal SP8192 base plus the current data-loader variants
- `submission.json`: experiment metadata
