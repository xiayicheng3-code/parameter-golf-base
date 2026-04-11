# Transformer State Prefetch v2

This run moves the grammar state machine out of the GPU forward path and adds stronger structural priors on both the data and model side.

## Idea

- Token shards are read through `np.memmap`.
- We recover document starts from the per-document BOS token.
- We train on document-local windows only. Each document is viewed as a "closed" sequence that starts at its own `BOS` and, when available, ends by predicting the next document's real `BOS`.
- CPU workers process documents independently.
- Each worker builds grammar feature ids for a whole document, slices them into fixed windows, and hands `numpy` batches plus loss masks to the main process through a `DataLoader`.
- The main process owns `torch.from_numpy(...)`, optional `pin_memory()`, and non-blocking H2D copies so the GPU mostly sees ready-made `input_ids`, `target_ids`, and `grammar_feature_ids`.
- The first few model layers can be restricted to chunked sliding-window causal attention plus causal depthwise convolution before later layers see full causal attention.
- The training stream is now effectively infinite, so persistent worker pools do not get torn down and rebuilt whenever one pass over the current document set finishes.

The model now learns:

```text
logits = transformer(tok_emb + pos_emb + embed(grammar_feature_ids))
loss = masked_xent(logits, targets, loss_mask)
```

The GPU no longer runs the serial grammar state update loop during training, and short documents are padded without corrupting the loss.

## Pipeline

```text
memmapped shards
-> BOS scan for doc boundaries
-> closed-doc view (predict next real BOS when available)
-> doc-level CPU workers
-> full-doc grammar feature build
-> train phase sampler / eval rolling scorer
-> numpy batch IPC
-> main-process pin_memory
-> non-blocking device prefetch
-> local-only early blocks (chunked sliding-window attention)
-> Transformer forward
```

## Current Notes

- Document boundaries are inferred from `bos_id`, which is present for every exported document.
- Documents may cross shard boundaries; the memmap token store stitches spans across files.
- Training windows use a two-phase non-overlapping sampler for long documents:
  - phase 0: `0, W, 2W, ...`
  - phase 1: `U-kW, U-kW+W, ...`
  - phase selection is deterministic per doc and flips each epoch.
- Local attention is now implemented as chunked sliding-window attention. Queries are processed in chunks and only materialize the nearby K/V span they can legally see, rather than building a full `T x T` local mask.
- Evaluation keeps the same document-local assumption but uses exact rolling windows: the first window scores all visible targets, and later windows stride by `64` while only scoring newly exposed target positions.
- Features are transferred as `uint8` and only cast to `long` at embedding lookup time.
- Token ids are shipped from workers as `uint16`, then cast on device before embedding / loss. This reduces host-to-device bandwidth pressure relative to `int64` batches.
- Workers emit `numpy` batches instead of `torch.Tensor` batches. This avoids `torch_shm_manager` / shared-memory-manager failures on some local macOS sandbox setups and keeps the main-process H2D logic explicit.
- Train workers cache reusable grammar feature tensors for moderate-size documents, so phase-flipped revisits do not recompute the entire serial state machine every time.
- A CUDA-only host-prefetch thread overlaps `numpy -> torch -> pin_memory` preparation with GPU compute. CPU local smoke tests may not speed up from this, but cloud GPUs should see fewer host-side bubbles.
- Worker processes clamp PyTorch intra-op threads to `1` by default to avoid CPU oversubscription when many grammar builders run in parallel.
- Eval no longer drops the last partial batch; every evaluation window is scored.
- This is a performance architecture scaffold, not a final leaderboard script.

## Local Smoke

Run from this folder:

```bash
RUN_ID=state_prefetch_v2_docphase_smoke \
DEVICE=cpu \
NUM_WORKERS=0 \
EVAL_NUM_WORKERS=0 \
TRAIN_TOKEN_LIMIT=50000 \
VAL_TOKEN_LIMIT=20000 \
TRAIN_DOC_LIMIT=64 \
VAL_DOC_LIMIT=16 \
SEQ_LEN=64 \
BATCH_SIZE=4 \
STEPS=3 \
D_MODEL=64 \
N_LAYERS=1 \
N_HEADS=4 \
LOCAL_ATTN_LAYERS=1 \
LOCAL_WINDOW=32 \
LOCAL_ATTN_CHUNK=16 \
LOCAL_CONV_KERNEL=5 \
EVAL_EVERY=0 \
/Users/yichengxia/ML_NN_DA/.venv/bin/python train_gpt.py
```

To verify the document-worker path:

```bash
RUN_ID=state_prefetch_v2_docphase_workers2 \
DEVICE=cpu \
NUM_WORKERS=2 \
EVAL_NUM_WORKERS=1 \
TRAIN_TOKEN_LIMIT=50000 \
VAL_TOKEN_LIMIT=20000 \
TRAIN_DOC_LIMIT=64 \
VAL_DOC_LIMIT=16 \
SEQ_LEN=64 \
BATCH_SIZE=4 \
STEPS=3 \
D_MODEL=64 \
N_LAYERS=1 \
N_HEADS=4 \
LOCAL_ATTN_LAYERS=1 \
LOCAL_WINDOW=32 \
LOCAL_ATTN_CHUNK=16 \
LOCAL_CONV_KERNEL=5 \
EVAL_EVERY=0 \
/Users/yichengxia/ML_NN_DA/.venv/bin/python train_gpt.py
```

## Smoke Results

These are architecture checks, not quality claims:

| Run | Purpose | `val_loss` | `val_bpb` | Artifact bytes | Total bytes | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `state_prefetch_v2_sliding_smoke` | single-process check for doc-only windows, loss masks, and chunked sliding local block | `6.939736` | `4.039225` | `257132` | `297166` | `22.137s` |
| `state_prefetch_v2_sliding_workers2` | multi-worker check for the same pipeline | `6.939060` | `4.038832` | `257160` | `297194` | `22.685s` |

Both used the same tiny CPU setup with `SEQ_LEN=64`, so they are architecture checks rather than quality claims. The multi-worker path runs correctly, but this smoke is still too short to show a throughput win. The expected benefit is on longer runs or CUDA jobs where GPU compute can overlap with CPU-side document feature preparation and where early local-only layers reduce unnecessary long-range attention work.
