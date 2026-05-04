# M52: LoRA Forward, Projected Muon Gradient

Fork of the April 9 legal SP8192 baseline for testing LoRA as a train-time gradient-direction sampler rather than as stored model capacity.

Core update:

```python
Y = X @ W.T
loss.backward()
# custom backward synthesizes the B gradient that the zero-B LoRA branch would have produced
W.grad = B.grad @ A
optimizer_W.step()
B.zero_()
B.grad = None
```

Implementation notes:

- `LORA_GP_ENABLED=1` by default in this experiment copy.
- `LORA_GP_RANK=128` is the first-screen default.
- `LORA_GP_REFRESH_EVERY=1` refreshes `A` every optimizer step.
- Each rank seeds `A` independently, so `B.grad` is intentionally not DDP-averaged.
- `A` and `B` are non-persistent buffers; export still serializes only the original model weights.
- The hot forward path does not materialize the zero-valued LoRA branch; custom autograd computes the projected low-rank gradient in backward.

First run:

```bash
LORA_GP_ENABLED=1 LORA_GP_RANK=128 LORA_GP_REFRESH_EVERY=1 torchrun --standalone --nproc_per_node=8 train_gpt_human.py
```
