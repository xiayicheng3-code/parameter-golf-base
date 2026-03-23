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
| M02 | native_sliding_window_attention | attention | baseline | since many SOTAs are doing sliding window style evaluation but not trained on, it's reasonable to do so on training, and this makes super long sequences that made TTT possible if modification needed | usable baseline | Flash Attention incompatible | GQA | XSA | me | locked |
| M03 | shifted attention | attention | single | a time related attention mechanism at lower levels could function as a skip-bigram | sliding window attention | May require Full attention or ultra long window to mimic TTT, which significantly reduces training efficiency. Causing incompatibility of some flash attention kernel | any GQA | bigram/ N-gram | me | candidate |
| M04 | causal deltanet | attention |single | a local gradient descending near linear component that could have learned a skip bi-gram during test time | Flash Attention kernel incompatibility, super long training time, gradient explosion, low performance at initial characters, require ultra long sequence to learn. | Linear attention | Sliding window attention, N-gram | me | frozen |
| M05 | delta compression from random | quantization | By utilizing random information, the average magnitude of recorded vectors is significantly reduced, and small values increased, which faciliates zip compression | Normal quantization | Barely compatible with QAT, may experience extreme quantization loss on outliers, works bad on INT 5, may results in size shifts on different seeds | None | any modification on FFN or attention, especially looping | me | candidate |
| M06 | space_factorized | Since space always appears before a token, maybe using additive embedding will make it efficient (-10%) | embedding | anti-bigram design, may reduce the effectiveness of embedding and bigrams | me | candidate
| M07 | simple hyper connection | redefine resnet as xn = f(rmsnorm( a_t-1*x_t-1 + ... a_0 * x_0)), with a_n initialized as 1 to mimic resnet |residule stream | single/ paired | given additional degree of freedom in residule stream, this low parameter mechanism helps transformer to utilize lower level information, but it's data irrelevent compare to other hyper connections, which makes training and semantic manifold stable | resnet | number of parameter may increase O(n^2) in larger model or looping transformers. Paramters may shift too far from 1 (but a wight decay centered at 1 may help, or use e^(a=0)) | None | me | candidate

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

| Run ID | Date | Baseline Ref | Change Set | Train Budget | Seed | Main Metric | Delta vs Base | Stability | Speed / Memory Note | Verdict | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S00 | 2026-03-23 | M01 | baseline repro | small | 42 |  |  |  |  | anchor | keep |
| S01 | 2026-03-23 | M01 | M02 only | small | 42 |  |  |  |  |  |  |
| S02 | 2026-03-23 | M01 | M03 only | small | 42 |  |  |  |  |  |  |
| S03 | 2026-03-23 | M01 | M04 only | small | 42 |  |  |  |  |  |  |

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
