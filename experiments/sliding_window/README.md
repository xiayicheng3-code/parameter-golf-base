# Sliding Window Experiments

This idea family is for training or evaluation schemes that use local/sliding attention windows rather than full causal attention over the entire sequence.

Recommended naming:

- `baseline_repro`
- `eval_stride64`
- `eval_stride32`
- `eval_multiscale_64_256`
- `train_window1024_seq4096`
- `train_window512_seq8192`

Recommended metadata to record for each run:

- training sequence length
- local attention window size
- evaluation stride
- whether the change is train-time, eval-time, or both
- steps completed within the wallclock cap
- step time
- pre-quant and post-quant `val_bpb`
- total artifact size

Suggested first sweep order:

1. Reproduce current baseline in this folder.
2. Keep training fixed and vary eval stride/windowing.
3. Try multiscale eval.
4. Try train-time local window attention with longer packed sequences.
5. Compare quality gain against throughput loss.
