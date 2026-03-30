# M34: Multi-Hash Gated Ngram Embedding

This experiment starts from the 11L `BigramHash + SmearGate` style baseline and
replaces the single-bucket bigram lookup with a flat-fused multi-order n-gram
embedding module.

Current mixing modes:

- `single`: compatibility mode matching the old single-candidate behavior
- `mean`: simple flat averaging over all n-gram hash candidates
- `scalar_gate`: context-conditioned scalar mixing over hash candidates
- `attn_lite`: tiny query-key selection over hash candidates, followed by an `[x; attn_out]` pass-through gate

Candidate-id execution modes:

- `NGRAM_CANDIDATE_SOURCE=inline`: build hash candidate ids inside the compiled model
- `NGRAM_CANDIDATE_SOURCE=gpu_eager`: build candidate ids outside the compiled model, but still on GPU
- `NGRAM_CANDIDATE_SOURCE=cpu`: build candidate ids on CPU and pass them in as an extra model input

Current recommendation:

- Use `NGRAM_CANDIDATE_SOURCE=inline` for actual training runs.
- Treat `gpu_eager` and `cpu` as debugging/profiling paths only. In our current experiments they were slower than `inline`, so they are kept only for bottleneck isolation.
- Treat the Triton prototype in [fused_ngram_single_io.py](/Users/yichengxia/ML_NN_DA/parameter_golf/parameter-golf-base/experiments/m34_multi_hash_gated/fused_ngram_single_io.py) as research-only for now. It is not integrated into `train_gpt.py` and should not be used for main experiments yet.

Hypothesis:

- Standard single-bucket bigram hashing wastes capacity on destructive collisions.
- Flat multi-order candidate fusion may let the model surface structurally important
  local token combinations without changing the tokenizer.
- Learned mixing may recover more of the correct local lexical feature without
  paying for a much larger explicit n-gram table.

Suggested local smoke command:

```bash
cd parameter-golf-base/experiments/m34_multi_hash_gated
RUN_ID=m34_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=65536 \
TRAIN_SEQ_LEN=512 \
EVAL_SEQ_LEN=512 \
ITERATIONS=20 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=5 \
BIGRAM_DIM=128 \
NGRAM_ORDERS=2,3,4 \
NGRAM_VOCAB_SIZES=4096,2048,1024 \
NGRAM_NUM_HASHES=2,2,2 \
NGRAM_MIX_MODE=scalar_gate \
NGRAM_INSERT_POS=after_smear \
NGRAM_CANDIDATE_SOURCE=inline \
python3 train_gpt.py
```

Suggested comparison ladder:

1. `NGRAM_ORDERS=2 NGRAM_VOCAB_SIZES=4096 NGRAM_NUM_HASHES=1 NGRAM_MIX_MODE=single`
2. `NGRAM_ORDERS=2,3,4 NGRAM_VOCAB_SIZES=4096,2048,1024 NGRAM_NUM_HASHES=2,2,2 NGRAM_MIX_MODE=mean`
3. `NGRAM_ORDERS=2,3,4 NGRAM_VOCAB_SIZES=4096,2048,1024 NGRAM_NUM_HASHES=2,2,2 NGRAM_MIX_MODE=scalar_gate`
4. `NGRAM_ORDERS=2,3,4 NGRAM_VOCAB_SIZES=4096,2048,1024 NGRAM_NUM_HASHES=2,2,2 NGRAM_MIX_MODE=attn_lite`
