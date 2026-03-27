# Polar Weight Probe

Quick CPU-side experiment for a polar-style weight codec.

This is a lightweight harness for checking whether a blockwise polar transform looks promising for model weight storage:

- load or build a model
- compress its weights with a simple blockwise polar codec
- decode the weights
- run one inference pass before and after decode
- report payload size and reconstruction / output error

It is intentionally exploratory, not submission-ready.

## What It Does

The codec works blockwise on flattened tensors:

1. Optionally apply a normalized Hadamard mix inside each block.
2. Convert each block into a binary tree of polar coordinates.
3. Store one root radius in `float16`.
4. Quantize first-layer angles over `[-pi, pi]`.
5. Quantize higher-layer angle residuals around `pi/4`.

The higher layers are where the "angles concentrate near 45 degrees" intuition can show up.

## Quick Start

The default path now targets a Hugging Face causal LM:

```bash
python3 experiments/polar_weight_probe/polar_weight_probe.py \
  --model-id distilgpt2 \
  --prompt "The quick brown fox jumps over"
```

If you want a fully local smoke test that uses only `numpy`:

```bash
./.venv/bin/python experiments/polar_weight_probe/polar_weight_probe.py --source toy
```

If `torch` and `torchvision` are installed, you can still probe a vision model:

```bash
python3 experiments/polar_weight_probe/polar_weight_probe.py \
  --source torchvision \
  --model-id resnet18 \
  --pretrained
```

If `torch` and `transformers` are installed, you can probe a Hugging Face causal LM:

```bash
python3 experiments/polar_weight_probe/polar_weight_probe.py \
  --source hf \
  --model-id distilgpt2 \
  --prompt "Parameter golf is about squeezing more quality into fewer bytes."
```

## Useful Knobs

- `--block-size 8` or `16`: block size for the polar tree
- `--first-level-bits 12`: precision for signed bottom-level angles
- `--upper-level-bits 8`: precision for upper angle residuals
- `--no-hadamard`: disable the fixed in-block mixing step
- `--json`: print only the JSON summary

## Notes

- The toy path is the only one smoke-tested in this repo right now.
- Real-model paths require local installs of `torch` plus either `torchvision` or `transformers`.
- This is aimed at weight-file compression experiments, not direct polar-domain inference.
