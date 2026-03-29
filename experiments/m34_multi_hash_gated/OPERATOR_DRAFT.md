# Streaming Softmax N-Gram Fusion Operator Draft

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

Use elementwise candidate-content logits:

```text
logit_i = q * v_i + b_i
```

Then compute a streaming softmax across candidate slots for each token and
feature dimension:

```text
m = max_i logit_i
den = sum_i exp(logit_i - m)
out_num = sum_i exp(logit_i - m) * v_i
out = out_num / den
```

Where:

- `q` is a token-conditioned query vector in the same dimension as the candidate
  embedding
- `v_i` is the candidate embedding for candidate `i`
- `b_i` is an optional learned per-slot bias

This preserves content-aware filtering while remaining compatible with a
single-read streaming kernel. It is not standard qk-attention, but it is still a
softmax-normalized content-sensitive fusion operator.

## Why This Is Single-Read Friendly

For each candidate `v_i`, the kernel can:

1. load `v_i`
2. compute `logit_i = q * v_i + b_i`
3. update running max / running denominator
4. update running weighted output numerator
5. discard `v_i`

No second candidate-value read is required by the math, as long as the operator
maintains online softmax state inside the kernel.

## Tensor Shapes

The prototype API uses flattened token shape:

- `q`: `[N, D]`
- `candidate_ids`: `[N, K]`
- `embed_weight`: `[V, D]`
- `slot_bias`: `[K]` optional
- output: `[N, D]`

Where:

- `N = batch * seq_len`
- `D = ngram_dim`
- `K = total_candidates`
- `V = total rows in the unified candidate embedding table`

## Kernel Strategy

One Triton program handles a tile of tokens and a tile of embedding dimensions.
For each candidate slot:

- load candidate ids for the token tile
- gather candidate embedding tile
- load query tile
- compute logits `q * v + bias`
- update running softmax max
- rescale previous accumulators
- update denominator and weighted numerator

At the end:

- write `out_num / out_den`

This is the same systems intuition as FlashAttention: keep normalization and
weighted accumulation in one streaming loop instead of separating score and value
passes in global memory.

## Prototype Scope

The prototype is intentionally limited:

- forward path: Triton when available
- backward path: PyTorch reference implementation
- no attempt yet to fuse hash generation
- no attempt yet to integrate directly into `train_gpt.py`

This keeps the prototype honest while still letting us test whether a
single-read softmax fusion operator is worth a full custom kernel.

## Open Questions

- whether elementwise softmax-normalized candidate filtering keeps the same
  step-wise learning benefit as the original attention-style filter
- whether slot bias alone is enough, or whether order/hash metadata should also
  modulate the logits
- whether logits should remain purely `q * v + b`, or whether an extra per-slot
  scale is worth the complexity
- whether the forward-only Triton speedup is already enough to justify a full
  backward kernel
