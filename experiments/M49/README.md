# M49: Bivariate Activation Sandbox

This experiment folder is a clean branch for two-input / gated FFN activations.

Initial setup:
- `train_gpt.py` is copied from `experiments/M48/train_gpt.py`
- `M48` remains the univariate activation line
- `M49` is reserved for bivariate activations such as `SwiGLU`-style baselines and follow-up gated variants
- current default is a simple `SwiGLU` baseline with `MLP_MULT=2.625`, i.e. hidden width `1344` at `model_dim=512`

Recommended workflow:
- First establish a simple `SwiGLU` baseline here
- Then compare parameter-neutral and wallclock-matched gated variants against the current `leaky_relu^2` record line
