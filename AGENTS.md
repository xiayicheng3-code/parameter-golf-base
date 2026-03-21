# AGENTS.md

This repository is for the OpenAI Model Craft Challenge: Parameter Golf.

Read [README.md](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/README.md) before making meaningful changes.

## Competition Constraints

- The objective is to minimize FineWeb validation bits per byte (`val_bpb`) under a strict artifact cap.
- The submission artifact limit is **16,000,000 bytes total**, not 16 MiB.
- Counted artifact size is: `train_gpt.py` code bytes + compressed model bytes.
- Leaderboard submissions must train reproducibly in **under 10 minutes on 8x H100 SXM GPUs**.
- Evaluation may also take up to 10 minutes on 8x H100s, but not longer.
- No network calls, external downloads, or training-data access are allowed during evaluation.
- The submitted artifact must be self-contained and reproducible.
- Validation is tokenizer-agnostic and based on FineWeb validation compression (`val_bpb`), not just cross-entropy loss.
- New SOTA submissions must beat the previous record by at least **0.005 nats** and include evidence strong enough to support `p < 0.01`, unless the change is purely systems optimization.
- If tokenizer or dataset handling changes, correctness of `val_bpb` must be demonstrated convincingly.
- OpenAI may disqualify submissions that violate the spirit of the challenge, including unfair use of external compute.

## Repo Expectations

- Baseline scripts such as [`train_gpt.py`](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/train_gpt.py) and [`train_gpt_mlx.py`](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/train_gpt_mlx.py) are starting points, not where final SOTA work should live.
- Strong experimental runs and final submissions belong under [`records/`](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/records), with their own `README.md`, `submission.json`, logs, and runnable training code.
- All experiments should start in [`experiments/`](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/experiments), with their own `README.md`, `submission.json`, logs, and runnable training code.
- PRs that improve the core scripts are acceptable only if they do not significantly increase complexity.
- Do not assume the MLX script is feature-equivalent to the CUDA script. It is a portability path for local iteration on Apple Silicon.

## Practical Guidance For Agents

- Optimize for measured `val_bpb` under the 16 MB artifact cap, not for parameter count alone.
- Treat compression format as part of the model design. Quantization, tying, low-rank structure, recurrence, and test-time compute are all in-scope if they fit the rules.
- Be careful with tokenizer or dataset edits. Small bugs there can produce invalid but superficially better scores.
- Preserve reproducibility. Record exact environment variables, commands, dataset variant, tokenizer path, and output artifact sizes.
- Assume final evaluation runs in the provided Runpod-style environment with dependencies preinstalled.
- For local smoke tests, smaller subsets are fine; for claims about results, use the challenge dataset setup described in the README.
- Prefer changes that keep the training script runnable from inside a submission folder without hidden external dependencies.
- If modifying logging or outputs, keep the final metrics easy to parse: `val_loss`, `val_bpb`, and compressed artifact size are essential.
- Avoid introducing runtime behavior that depends on internet access, mutable external state, or unavailable local files.

## Submission Checklist

- Include a per-run `README.md` explaining the idea and setup.
- Include `submission.json` with required metadata.
- Include the produced training log.
- Include a runnable `train_gpt.py` and any required local dependencies in the submission folder.
- Make sure the script actually runs from the submission directory.
- For non-record or unlimited-compute runs, state that clearly in the submission README.

## Local Workflow Notes

- Apple Silicon users can iterate with [`train_gpt_mlx.py`](/Users/yichengxia/神经网络_机器学习/parameter_golf/parameter-golf-base/train_gpt_mlx.py).
- Serious leaderboard validation should be checked against the CUDA path and the actual 8xH100 constraint.
- Default baseline behavior includes a wallclock cap; do not accidentally compare runs with different wallclock settings.
- Periodic validation may be disabled depending on script and environment variables; verify what is actually being measured before drawing conclusions.

## When Unsure

- Default to the spirit of the challenge: self-contained, reproducible, size-constrained, and honestly comparable runs.
- If a change could affect score validity, artifact accounting, or reproducibility, document it explicitly.
