# Experiment Matrix

This file is the cheap-decision dashboard for architecture changes.

Use it to answer four questions:

| Question | What to record |
| --- | --- |
| Is the module worth more budget? | Small-budget win/loss/stability |
| Does it need a partner? | Dependency and complementary module |
| Is it dangerous to mix? | Conflict risk and failure mode |
| What do we do next? | `continue` / `freeze` / `retest` / `pair-only` |

## 1. Module Inventory

| ID | Module | Area | Type | Hypothesis | Expected Win | Cost / Risk | Depends On | Likely Conflicts | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | Baseline | full model | baseline | Fixed reference point for all comparisons | Stable ranking anchor | None | None | None | me | locked |
| M02 | native_sliding_window_attention | attention | baseline | since many SOTAs are doing sliding window style evaluation but not trained on, it's reasonable to do so on training, and this makes super long sequences that made TTT possible if modification needed | usable baseline | Kernel / implementation constraints depend on the exact flash-style attention backend and windowing strategy | GQA | XSA | me | locked |
| M03 | shifted attention | attention | single | a time related attention mechanism at lower levels could function as a skip-bigram | sliding window attention | May require Full attention or ultra long window to mimic TTT, which significantly reduces training efficiency. Causing incompatibility of some flash attention kernel | any GQA | bigram/ N-gram | me | candidate |
| M04 | causal deltanet | attention | single | a local gradient descending near linear component that could have learned a skip bi-gram during test time | Can add adaptive local memory beyond vanilla attention | Flash Attention kernel incompatibility, super long training time, gradient explosion, low performance at initial characters, may require ultra long sequence | linear attention kernels | native sliding window attention, N-gram paths | me | frozen |
| M05 | delta compression from random init | quantization | By storing weights as deltas from the initial state ( using definite random algorithm initialization), average magnitudes shrink and small values become more common, which should help entropy coding | Better zip / zlib friendliness under fixed artifact cap | Barely compatible with QAT, can amplify low-bit outlier damage, unstable at INT5, may shift compressed size across seeds | None | FFN or attention changes that create heavy outliers, especially looping | me | candidate |
| M06 | space_factorized embedding | embedding | paired | Since space almost always appears before a token, an additive space factor may encode this cheaply | Possible small embedding efficiency win | Anti-bigram bias may reduce usefulness of bigram features | token embedding path | bigram hash, smear-style embedding tricks | me | candidate |
| M07 | simple hyper connection | residual stream | paired | Redefine residual stream as a learned weighted combination of earlier states so lower-level signals survive longer | Cheap extra routing flexibility | Parameter count can scale poorly with depth; weights may drift far from identity without regularization | residual backbone | looping / recurrent residual designs | me | candidate |
| M08 | sliding_window_eval | evaluation | single | Overlapping evaluation windows recover context that non-overlapping chunks throw away | Reliable eval-only BPB gain | Slower eval; can hide weak train-time context if compared carelessly | standard causal LM | train-time long-context claims | public | survived |
| M09 | FP16_tied_embedding_export | quantization | single | Keeping tied embedding / output head in FP16 protects the most quantization-sensitive weights | Better post-quant quality at modest size cost | Uses part of the 16 MB budget; reduces room for other parameters | tied embeddings | aggressive mixed quant budgets | public | survived |
| M10 | tuned_lower_lr | optimization | single | Default learning rates are too high for the 10-minute regime; lower or retuned LR should trade step quality for better final score | Better convergence under fixed wallclock | Overtuning may undertrain if other modules change throughput | None | aggressive warmdown or very deep models | public | survived |
| M11 | warmdown_schedule_fix | optimization | single | Warmdown should be aligned to actual wallclock training length instead of a mismatched default step count | Better late-stage convergence and quantization readiness | Too much decay can undertrain; schedule interacts with throughput | LR schedule | heavy QAT overhead | public | survived |
| M12 | extra_depth_10L | architecture | single | Compression-aware training can buy enough size budget to fit an extra layer profitably | Better capacity under same artifact cap | More compute per step may erase gains if throughput drops too much | stable optimizer / compression path | slow activations, expensive recurrent modules | public | survived |
| M13 | mixed_int6_int8_quant | quantization | single | Sensitive tensors should stay at higher precision while compressible tensors go lower bit | Better size/quality frontier than uniform quantization | Requires good tensor partitioning; easy to over-compress embeddings or heads | export pipeline | delta compression, naive uniform INT5/INT6 | public | survived |
| M14 | int6_qat_ste | quantization | paired | Fake-quant during training can remove quantization gap and make lower-bit export usable | Stronger post-quant robustness | Training overhead can reduce step count; requires careful integration | quantized export path | very slow models, unstable custom kernels | public | survived |
| M15 | MLP_3x_expansion | FFN | single | Wider FFN may be a better use of budget than more attention complexity | Better token mixing / capacity | Higher parameter count and slower step time | stable compression scheme | already-large embeddings, expensive gating | public | survived |
| M16 | muon_weight_decay | optimization | paired | Decoupled weight decay on Muon can regularize magnitudes and make compression easier | Better generalization and quantization friendliness | Interacts with LR, momentum, and model depth | Muon optimizer | non-Muon baselines, very weak init | public | survived |
| M17 | higher_muon_momentum_warmup | optimization | paired | Higher Muon momentum with warmup may improve late-run efficiency in the 10-minute budget | Better sample efficiency | Can destabilize early steps if combined with high LR | Muon optimizer | poorly tuned LR, fragile recurrent modules | public | candidate |
| M18 | SmearGate | embedding | single | A learned previous-token blend can inject cheap bigram context before attention | Better local context at tiny parameter cost | May overlap with other bigram modules and complicate quantization | token embedding path | shifted attention, other explicit bigram modules | public | survived |
| M19 | BigramHash_embedding | embedding | paired | A hashed bigram table can add token-pair features cheaply enough to fit the cap | Strong local lexical prior | Collisions and extra table size; may overfit to frequent pairs | token embedding path | shifted attention, space factorization | public | survived |
| M20 | orthogonal_init | initialization | paired | Orthogonal init may accelerate convergence and pair naturally with Muon-style updates | Better early optimization | Benefit may vanish without matching optimizer dynamics | Muon or stable deep stack | arbitrary init-sensitive modules | public | survived |
| M21 | SWA_late_checkpoint_avg | optimization | paired | Averaging late checkpoints can smooth the weight distribution and help quantization | Better post-quant robustness | Extra bookkeeping; bad averaging window can blur useful specialization | stable late training | unstable or still-improving late runs | public | survived |
| M22 | overtone_init | initialization | paired | Frequency-aware or overtone-style init may seed better local structure for language modeling | Possible faster early learning | Mechanism is less standard and may be brittle across scripts | custom init path | orthogonal init assumptions | public | candidate |
| M23 | SwiGLU | FFN | single | SwiGLU can outperform ReLU-style FFNs if throughput loss is affordable | Better per-step modeling quality | Often slower, so wallclock-limited runs may lose total steps | FFN path | extra depth, long-sequence runs | public | candidate |
| M24 | LoRA_TTT | test-time adaptation | interaction | Per-document LoRA adaptation can recover context-specific performance without changing the frozen backbone much | Strong eval-time gain if allowed by budget | High eval complexity and workflow cost; must avoid leakage | eval-time adaptation loop | strict no-adaptation baselines, long eval windows | public | candidate |
| M26 | delta_lowbit_float_export | quantization | paired | delta weights may be better matched by low-bit floating-point formats than linear INT5/INT6 because they likely have many near-zero values plus a few critical outliers | better post-quant quality at similar compressed size for delta-based exports | custom serialization and roundtrip complexity; may not beat int formats after entropy coding; format choice still unresolved | delta compression from random init | mixed_int6_int8_quant, naive uniform low-bit integer quantization | me | pair-only |
| M25 | looping_layer_RoPE | architecture | interaction | Reuse a 3-layer core across repeats, inject pass identity through MLP-only LayerRoPE, and let per-pass LoRA absorb the attention and FFN specialization gap | Could buy more effective depth per byte without fully collapsing shared layers | Shared weights may underfit or quantize poorly; full LoRA coverage can erase the budget win if rank is too high | stable optimizer, careful quantization readout | hard layer sharing, aggressive low-bit export | me | candidate |

Type guide:

| Type | Meaning |
| --- | --- |
| `single` | Should have a visible effect by itself |
| `paired` | Probably only makes sense with another module |
| `interaction` | Mainly interesting because of cross-term effects |
| `baseline` | Fixed comparison target |

Status guide:

| Status | Meaning |
| --- | --- |
| `candidate` | Not screened yet |
| `screening` | Running or queued for cheap test |
| `survived` | Worth extra budget |
| `pair-only` | Not useful alone, but still viable in combos |
| `frozen` | Park for now |
| `locked` | Baseline; do not move casually |

## 2. Cheap Screening Board

Keep this table dense. One row = one low-budget decision point.

| Run ID | Date | Baseline Ref | Change Set | Train Budget | Seed | BPB | final bytes | Stability | Speed: total token used | Verdict | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S00 | 2026-03-23 | M01 | baseline repro | small | 42 |  |  |  |  | anchor | keep |
| S01 | 2026-03-23 | M01 | M02 only | small | 42 |  |  |  |  |  |  |
| S02 | 2026-03-23 | M01 | M03 only | small | 42 |  |  |  |  |  |  |
| S03 | 2026-03-23 | M01 | M04 only | small | 42 |  |  |  |  |  |  |
| S04 | 2026-03-23 | M01 | M25 only | small | 42 |  |  |  |  |  |  |

Verdict vocabulary:

| Verdict | Meaning |
| --- | --- |
| `promising` | Better enough to keep investing |
| `flat` | No meaningful gain |
| `unstable` | Too noisy or fragile |
| `regression` | Worse than baseline |
| `needs-pair` | Weak alone but theory still alive |

## 3. Pair / Interaction Matrix

Mark only the combinations you actually care about.

| Left \ Right | M02 | M03 | M04 | Notes |
| --- | --- | --- | --- | --- |
| M02 | self |  |  |  |
| M03 |  | self |  |  |
| M04 |  |  | self |  |

Suggested marks:

| Mark | Meaning |
| --- | --- |
| `++` | Strong theoretical complement |
| `+` | Worth trying |
| `0` | No clear interaction story |
| `-` | Some overlap / possible cannibalization |
| `--` | High conflict risk |
| `?` | Intuition says interaction exists, but unclear direction |

## 4. Priority Funnel

This is the table you look at before spending H100 money.

| Rank | Candidate | Why It Is Still Alive | Evidence Level | Recommended Budget | Decision Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | M02 |  | weak | medium | If next run is not clearly positive, freeze |
| 2 | M03 + M04 |  | weak | medium | Only continue if pair beats best single |
| 3 | M04 |  | weak | small | If solo stays flat, relabel as `pair-only` |

Evidence level guide:

| Level | Meaning |
| --- | --- |
| `weak` | Single cheap run or incomplete signal |
| `medium` | Multiple runs agree on ranking |
| `strong` | Repeated win under near-real settings |

## 5. Run Log Index

Link every row to a real folder or note so you can backtrack later.

| Run ID | Path | Command | Notes |
| --- | --- | --- | --- |
| S00 | `baseline_repro/` |  |  |
| S01 |  |  |  |
| S02 |  |  |  |
| S03 |  |  |  |

## 6. Decision Notes

Use short bullet points only when the tables are not enough.

- Example: `M04` probably changes optimization dynamics more than representation quality.
- Example: `M02 + M03` may be redundant because both try to improve local token mixing.
- Example: if a module needs a custom LR or init to survive, treat that as part of its cost.
