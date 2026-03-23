# Looping Layer RoPE Prototype

Hypothesis:
- Replace a plain 9-layer stack with a `3 shared layers x 3 repeats` looping core.
- Give each repeat its own LoRA adapters so shared weights do not have to carry all pass-specific behavior.
- Put layer identity only on the MLP `W_up` path via a token-position-independent `LayerRoPE`.
- Apply LoRA after the `LayerRoPE`, so each repeat adapts inside its own rotated MLP input space.

What this prototype does:
- Keeps the virtual-space embedding experiment as the input/output path, so only the transformer trunk changes.
- Uses `NUM_PRELUDE_LAYERS + NUM_LOOP_LAYERS * LOOP_REPEATS + NUM_EPILOGUE_LAYERS == NUM_LAYERS`.
- Defaults to `NUM_PRELUDE_LAYERS=0`, `NUM_LOOP_LAYERS=3`, `LOOP_REPEATS=3`, `NUM_EPILOGUE_LAYERS=0`, giving an effective depth of 9.
- Reuses the same 3 core blocks across repeats.
- Attaches repeat-specific LoRA modules to `q/k/v/o` and `W_up/W_down`.
- Rotates only the MLP input-side columns of `W_up`, ignoring token position entirely.

Important implementation choices:
- `LayerRoPE` is a learned pairwise rotation over feature channels, not a token-position encoding.
- `LayerRoPE` is folded into the `W_up` weight path instead of being applied to activations directly.
- `LoRA` on `W_up` is also evaluated in the rotated space, matching the design choice that LoRA comes after the layer rope.
- Prelude and epilogue layers are supported, but disabled by default while we screen the clean 3x3 loop first.

Suggested smoke command:

```bash
cd parameter-golf-base/experiments/looping_layer_RoPE
RUN_ID=loop_rope_smoke \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_PATH=./fineweb_1024_bpe.vocab \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
NUM_PRELUDE_LAYERS=0 \
NUM_LOOP_LAYERS=3 \
LOOP_REPEATS=3 \
NUM_EPILOGUE_LAYERS=0 \
LORA_RANK=8 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
python3 train_gpt.py
```

Key things to measure next:
- whether the looping trunk keeps pre-quant quality competitive with a plain 9-layer stack
- whether per-repeat LoRA stabilizes shared blocks enough to reduce collapse across repeats
- whether `LayerRoPE` on `W_up` helps post-quant robustness or instead creates new quantization-sensitive directions
- whether a tiny prelude/epilogue is worth reintroducing after the pure loop core is screened
