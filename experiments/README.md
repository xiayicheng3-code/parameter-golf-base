# Experiments

This folder is for active iteration that should not live under `records/` yet.

Use one subfolder per idea family, and one nested subfolder per concrete run.

## Experiment Matrix

| Experiment | Status | Core idea | Baseline / control | Notes |
| --- | --- | --- | --- | --- |
| `web_grammar/transformer_state_only_v1` | Smoke verified scaffold | Feed 32 hand-authored cross-token web/prose/code grammar state features into a Transformer as input context only; the state machine should not provide a count-table base distribution. | Compare against a normal token-only tiny Transformer and the old `rule_count_hybrid_v1` count-table prototype if restored. | This is intentionally not the deleted/misnamed `transformer_grammar_hybrid_v1` residual setup. Avoid `combined = base_lp + logit_bias`; the model should learn `logits = f(grammar_state_window, token_context)` directly. Current risk: stateful feature loop is portable but needs vectorization/compilation before a serious graft. |
| `web_grammar/transformer_state_prefetch_v2` | Smoke verified scaffold | Move the 32-state grammar machine to CPU-side document workers, train on document-local BOS-aware windows, and optionally gate the first layers through chunked sliding-window attention plus causal depthwise conv. | Compare against `transformer_state_only_v1` for quality, then compare CUDA throughput with `NUM_WORKERS=0` vs `NUM_WORKERS>0`, `HOST_PREFETCH_BATCHES=0` vs `>0`, and `LOCAL_ATTN_LAYERS=0` vs `>0`. | Long docs use deterministic two-phase non-overlapping sampling that flips each pass. Eval uses exact rolling document-local scoring with masked loss. The cloud-oriented path now keeps train workers alive across cycles, caches moderate-size grammar features in workers, ships token ids as `uint16`, uses a CUDA host-prefetch thread to reduce pipeline bubbles, and avoids building full `T x T` local attention masks. |
| `web_grammar/sp8192_legalttt_grammar_port_v1` | Grafted onto 2026-04-09 legal SOTA | Start from the April 9 SP8192 legal record, keep its recurrence/parallel-residual/QK-gain/FA3/GPTQ stack, and transplant our web-grammar priors as a patch layer instead of rewriting the whole model. | First compare against the untouched `records/.../2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`, then ablate `DOC_LOCAL_WINDOWS`, `USE_GRAMMAR`, and `LOCAL_ATTN_LAYERS`. | Ported sub-ideas: `1.` document-local BOS-aware windowing with deterministic two-phase long-doc coverage, `2.` CPU grammar workers with feature caching and explicit host/device prefetch, `3.` input-only grammar adapter attached at the embedding layer, `4.` optional early chunked sliding-window attention plus causal depthwise conv while later layers stay on FlashAttention-3. Current caveat: document-local TTT is not ported yet, so this experiment should run with `TTT_ENABLED=0`. |

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
