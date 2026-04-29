# M26: Multi-Scale Leaky ReLU2 Temporal FFN

This experiment is copied from the April 9 SP8192 legal SOTA script and changes only the FFN input path.

Default graft:

- `M26_KERNELS=16,3,2,1`
- `M26_CHANNEL_SPLITS=256,128,64,64`
- `M26_LEAKY_SLOPE=0.5`

Each split gets a causal depthwise temporal filter initialized as an identity current-token filter. The recombined channels then pass through the baseline `LeakyReLU(0.5)^2` FFN, so the run starts close to the shipped baseline while giving the FFN a cheap learnable temporal pre-mix.

Run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
