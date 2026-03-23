# Shared Causal Delta + Sliding Attention Prototype

Hypothesis:
- Small models do not benefit much from full-sequence attention, so train-time attention should be local.
- Replacing full attention with sliding-window attention allows much longer training sequences while keeping attention cost linear in sequence length for fixed window size.
- Insert one shared causal delta module between attention and FFN so every layer reuses the same recurrent sequence-update rule.
- Keep FFNs and attention projections per-layer so the model still has local specialization capacity.

What this prototype changes:
- Makes `ATTENTION_IMPL=sliding_gqa` the default in this experiment script.
- Replaces full causal attention with chunked local causal attention:
  - each query chunk only attends to the trailing `SLIDING_WINDOW_SIZE` tokens
  - compute scales as `O(n * window)` for fixed window size
- Adds one shared `DualCausalDelta` module for the whole network:
  - `up` branch writes content
  - `gate` branch modulates with `SiLU`
  - key stream is built from the previous-step query via `k_t = q_{t-1}`
- Inserts the shared delta inside every block as:
  - `x = x + attn`
  - `x = x + shared_delta`
  - `x = x + mlp`
- Keeps baseline optimizer split, quantized serialization, and `val_bpb` accounting unchanged.

Important caveats:
- This script requires `flash-linear-attention` with `fla.ops.delta_rule.chunk_delta_rule` available in the runtime.
- The local workspace does not currently have that package installed, so this experiment has not been executed here.
- The delta module is shared, but each block has its own `delta_scale` and `delta_norm`, so the insertion is not fully tied.
- `SLIDING_WINDOW_SIZE` controls attention context, while `TRAIN_SEQ_LEN` can be increased independently.

Suggested smoke command:

```bash
cd parameter-golf-base/experiments/causal_deltanet/dual_causal_ttt
RUN_ID=shared_delta_slide_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=2048 \
ATTENTION_IMPL=sliding_gqa \
SLIDING_WINDOW_SIZE=512 \
SLIDING_CHUNK_SIZE=512 \
USE_SHARED_DELTA=1 \
DELTA_USE_QK_NORM=1 \
DELTA_USE_OUTPUT_PROJ=0 \
python3 train_gpt.py
```

Suggested remote 8xH100 starting point:

```bash
RUN_ID=shared_delta_slide_8xh100_v0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
ATTENTION_IMPL=sliding_gqa \
SLIDING_WINDOW_SIZE=512 \
SLIDING_CHUNK_SIZE=512 \
USE_SHARED_DELTA=1 \
DELTA_USE_QK_NORM=1 \
DELTA_USE_OUTPUT_PROJ=0 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Exact command for the longer-sequence 8xH100 run discussed in this workspace:

```bash
cd /workspace/parameter-golf/experiments/causal_deltanet/dual_causal_ttt
./prepare_data.sh
RUN_ID=shared_delta_slide_seq10240_v0 \
TRAIN_SEQ_LEN=10240 \
TRAIN_BATCH_TOKENS=655360 \
TRAIN_LOG_EVERY=100 \
./run_8xh100_seq10240.sh
```

Notes:
- `prepare_data.sh` runs from the repo root and downloads the published `sp1024` dataset plus tokenizer into `data/`.
- `run_8xh100_seq10240.sh` assumes you launch it from this experiment folder and uses paths relative to this folder.
- Override `NPROC_PER_NODE` if you want to smoke-test on fewer GPUs first, for example `NPROC_PER_NODE=1 ./run_8xh100_seq10240.sh`.

Key things to measure next:
- throughput vs full-attention baseline at matched `TRAIN_SEQ_LEN`
- whether longer `TRAIN_SEQ_LEN` actually helps once attention context is capped at `SLIDING_WINDOW_SIZE`
- whether the shared delta improves `val_bpb` over sliding attention alone
- whether a narrower or grouped shared delta can free enough bytes to increase `MODEL_DIM`
