# Streaming Softmax N-Gram Fusion Operator Draft

## Status

- This is a research draft, not a production path for current `m34` runs.
- Do not switch real experiments over to this Triton path yet.
- The current training script still uses the PyTorch `attn_lite` implementation because this prototype has not been end-to-end validated for integration, training speed, or backward cost.

## Motivation

The current `attn_lite` branch learns quickly per step, but the implementation is
too slow because it effectively performs a multi-stage consume of candidate
embeddings:

1. read candidate embeddings to compute scores
2. materialize intermediate score / normalization state
3. reuse candidate embeddings again to compute the weighted reduction

The problem is not necessarily softmax itself. The problem is that the current
high-level implementation gives no guarantee that score computation and value
reduction stay in SRAM in one streaming pass.

This draft keeps softmax-like filtering, but changes the operator target so the
kernel can consume each candidate embedding exactly once per tile.

## Design Goal

For each token and each candidate embedding `v_i`, we want:

- exactly one semantic read of `v_i`
- immediate score computation from the token-conditioned query and `v_i`
- online softmax statistics update
- immediate weighted accumulation into the output state

The key optimization target is IO flow, not replacing softmax with a different
activation.

## Chosen Prototype

Match the current `attn_lite` math directly:

```text
score_i = (q · k_i) / sqrt(d_attn) + b_i
alpha = softmax(score over candidate slots)
out = sum_i alpha_i * v_i
```

Where:

- `q` is a token-conditioned query vector
- `k_i` is the candidate key for candidate `i`
- `v_i` is the candidate value embedding for candidate `i`
- `b_i` is an optional learned per-slot bias

The prototype therefore uses separate key and value tables:

- `key_weight`: `[V, d_attn]`
- `value_weight`: `[V, d_value]`

This matches the core attention reduction in `train_gpt.py`. The pass-through
gate that follows `attn_lite` remains outside the fused operator.

## Why This Is Single-Read Friendly

For each candidate `(k_i, v_i)`, the kernel can:

1. load `k_i` and compute the scalar score `q · k_i`
2. load the needed tile of `v_i`
3. update running max / running denominator
4. update running weighted output numerator
5. discard this candidate

No intermediate score matrix needs to be materialized in HBM, and no separate
"softmax pass" over candidate scores is required, as long as the operator
maintains online softmax state inside the kernel.

## Tensor Shapes

The prototype API uses flattened token shape:

- `q`: `[N, A]`
- `candidate_ids`: `[N, K]`
- `key_weight`: `[V, A]`
- `value_weight`: `[V, D]`
- `slot_bias`: `[K]` optional
- output: `[N, D]`

Where:

- `N = batch * seq_len`
- `A = attn_dim`
- `D = ngram_dim`
- `K = total_candidates`
- `V = total rows in the unified candidate embedding table`

## Kernel Strategy

One Triton program handles a tile of tokens and a tile of value dimensions.
For each candidate slot:

- load candidate ids for the token tile
- gather the candidate key and compute the scalar attention score `q · k_i`
- gather the candidate value tile
- update running softmax max / denominator
- rescale previous accumulators
- update the weighted output numerator

At the end:

- write `out_num / out_den`

This is the same systems intuition as FlashAttention: keep score computation,
normalization, and weighted accumulation in one streaming loop instead of
separating them into multiple global-memory passes.

## Prototype Scope

The prototype is intentionally limited:

- forward path: Triton when available
- backward path: PyTorch reference implementation
- no attempt yet to fuse hash generation
- no attempt yet to integrate directly into `train_gpt.py`

This keeps the prototype honest while still letting us test whether a true
candidate-level attention fusion operator is worth a full custom kernel.

## Open Questions

- whether a separate learned key table is better than deriving keys from the
  value embedding table at runtime
- whether slot bias alone is enough, or whether order/hash metadata should also
  modulate the logits
- whether the key path should stay purely `q · k + b`, or whether a lightweight
  per-slot scale is worth the complexity
- whether the forward-only Triton speedup is already enough to justify a full
  backward kernel
