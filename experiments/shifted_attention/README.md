# Shifted Attention Experiments

This idea family is for exact local causal attention plus shifted-key variants that
simulate skip-bigram matching without changing the value stream.

What this baseline script changes:

- Starts from the repo `train_gpt.py` baseline instead of the `causal_deltanet` experiment.
- Keeps the transformer stack otherwise standard: no shared delta module and no extra recurrence.
- Adds `ATTENTION_IMPL` modes:
  - `gqa`: baseline full causal attention
  - `sliding_gqa`: exact local causal sliding-window attention
  - `shifted_gqa`: same exact local window, but selected copied KV heads use shifted keys
- Makes `sliding_gqa` the default mode in this experiment script.
- Uses `INIT_IMPL=stateless_ortho_v1` by default so large weight matrices can be regenerated from `INIT_SEED` without saving an init checkpoint.
- The default compression sweep is `raw_int8,delta_int8,raw_mixed,delta_mixed`. Override with `COMPRESSION_SCHEMES=...` to run only a subset.

Why this sliding implementation is the new base:

- The attention window is aligned to each query position, not to the chunk right edge.
- Query `t` can attend exactly to keys in `[t - window + 1, t]`.
- Chunking is only used to keep compute linear in sequence length for fixed window and chunk size.
- This makes the script a cleaner place for future attention modifications.

Shifted attention details:

- KV heads are expanded to query-head granularity before shifting.
- `V` stays in place.
- Selected copied `K` heads are shifted backward in time, so position `t` reads `K_{t-n}`.
- The default shifted pattern uses the first 3 layers with offsets `1,2,3,4`.
- No extra per-head mask is used for the shifted heads; the shared sliding causal mask is reused.
- Set `INIT_IMPL=legacy` if you want to compare against the old initialization path.

Suggested smoke command from this folder:

```bash
cd parameter-golf-base/experiments/shifted_attention
RUN_ID=exact_sliding_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=2048 \
ATTENTION_IMPL=sliding_gqa \
SLIDING_WINDOW_SIZE=512 \
SLIDING_CHUNK_SIZE=128 \
python3 train_gpt.py
```

Suggested shifted-attention smoke command:

```bash
cd parameter-golf-base/experiments/shifted_attention
RUN_ID=shifted_attn_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=2048 \
ATTENTION_IMPL=shifted_gqa \
SLIDING_WINDOW_SIZE=512 \
SLIDING_CHUNK_SIZE=128 \
SHIFTED_ATTENTION_LAYERS=3 \
SHIFTED_ATTENTION_OFFSETS=1,2,3,4 \
python3 train_gpt.py
```

Things to compare next:

- exact sliding vs full GQA at matched wallclock
- exact sliding vs the older chunk-right-edge approximation
- shifted heads only in early layers vs all layers
- different shifted offset sets, such as `1,2` or `1,2,4,8`
