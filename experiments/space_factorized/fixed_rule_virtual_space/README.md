# Fixed-Rule Virtual Space Prototype

Hypothesis:
- Reuse one base embedding table for paired SentencePiece pieces `piece` and `▁piece`.
- Represent the spaced form as `base_embedding[piece] + space_offset`.
- Keep the training target as a flat softmax over a virtual vocabulary, but only serialize the compact parameters.

What this prototype does:
- Starts from the existing `fineweb_1024_bpe.vocab`.
- Detects exact `▁piece` / `piece` pairs with a fixed rule.
- Collapses those pairs into one shared base entry plus one global `space_offset` vector.
- Remaps shard token ids on load from original SP ids into the virtual factorized ids.
- Uses tied embeddings only.

Current fixed-rule stats on `fineweb_1024_bpe.vocab`:
- original vocab: 1024
- paired `▁piece` / `piece` tokens: 119
- compact base vocab after folding: 905
- virtual vocab used for training logits: 1024

Important limitations:
- This is not a retrained tokenizer. It only folds pairs already present in the published 1024-piece vocab.
- Only exact pairs are folded. `▁foo` without bare `foo` stays untouched.
- The spaced/non-spaced difference is forced to be one global additive vector, which is a strong modeling constraint.
- Untied embeddings are not supported in this prototype.

Suggested smoke command:

```bash
cd parameter-golf-base/experiments/space_factorized/fixed_rule_virtual_space
RUN_ID=virtual_space_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_PATH=./fineweb_1024_bpe.vocab \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
python3 train_gpt.py
```

Key things to measure next:
- whether the 905-row compact embedding compresses materially better
- whether the global `space_offset` hurts loss too much
- whether a low-rank or per-class space offset is a better next step
