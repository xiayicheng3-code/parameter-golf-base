# M49: Bivariate Activation Sandbox

This experiment folder is a clean branch for two-input / gated FFN activations.

Initial setup:
- `train_gpt.py` is copied from `experiments/M48/train_gpt.py`
- `M48` remains the univariate activation line
- `M49` is reserved for bivariate activations such as `SwiGLU`-style baselines and follow-up gated variants
- current default is a grouped bivariate KAN-style variant on top of the `SwiGLU` split
- plain `SwiGLU` is still available with `MLP_ACTIVATION=swiglu`
- default `MLP_MULT=2.625`, i.e. hidden width `1344` at `model_dim=512`
- default grouped KAN settings use `KAN_GROUP_COUNT=16` and shared piecewise-linear `phi(u)` / `psi(g)` parameters per group
- when saving the full-precision model, `swiglu_kan` also writes `final_model.kan_shapes.pt` with grouped KAN parameters plus per-group `u/g` activation statistics

Recommended workflow:
- First compare the grouped `swiglu_kan` branch against the existing plain `SwiGLU` baseline
- Then decide whether richer bivariate shapes are helping before trying even less factorized two-input functions
