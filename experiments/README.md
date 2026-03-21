# Experiments

This folder is for active iteration that should not live under `records/` yet.

Use one subfolder per idea family, and one nested subfolder per concrete run.

Suggested layout:

```text
experiments/
  sliding_window/
    README.md
    baseline_repro/
    eval_stride64/
    eval_multiscale/
    train_local_window_2048/
```

Per-run checklist:

- `README.md` with the hypothesis and exact command
- `train_gpt.py` if the code differs from the repo baseline
- `train.log`
- any extra notes on artifact size, throughput, and validation method

Promotion rule:

- keep exploratory work here
- copy only the strongest, cleanest, reproducible runs into `records/`
