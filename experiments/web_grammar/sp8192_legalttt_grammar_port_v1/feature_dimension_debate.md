# SP8192 Grammar Feature Debate

Status: synthesized from a two-sided subagent debate plus local audit.

This note records a structured debate over the current output feature dimensions in
[`grammar_features.py`](/Users/yichengxia/ML_NN_DA/parameter_golf/parameter-golf-base/experiments/web_grammar/sp8192_legalttt_grammar_port_v1/grammar_features.py),
using the targeted `SP8192` audit pack in `/tmp/sp8192_audit_v2/sample_state_dump.json`.

Debate format:
- `Keep/Expand` side argues what each feature should do and how to improve it.
- `Prune/Merge` side argues what is redundant, misleading, or too weak to justify its slot.
- Final ruling records what we should keep, merge, simplify, redesign, or add next.

## Final Principles

- Keep parser-like states that encode truly serial, local, order-sensitive structure.
- Be skeptical of composite summary features that can already be derived from stronger base states.
- Prefer feature names that match actual semantics; if a feature is shallow, do not let its name overclaim.
- When in doubt, redesign weak features toward short-range transition cues rather than adding more broad global modes.
- The best next features are usually “what just happened?” and “what token comes right after that boundary?”, not deeper pseudo-grammar trees.

## Feature Rulings

| # | Feature | Keep/Expand View | Prune/Merge View | Final Ruling |
| --- | --- | --- | --- | --- |
| 0 | `line_state` | Strong BOS/line-start cue | Not redundant | Keep |
| 1 | `indent_bucket` | Useful coarse layout depth | Not redundant | Keep |
| 2 | `mode` | Valuable coarse regime selector | Heuristic and conflict-prone | Keep, but tighten precedence |
| 3 | `html_phase` | Good local tag/attr phase | Strongly justified | Keep |
| 4 | `json_depth` | Valuable nesting signal | Strongly justified | Keep |
| 5 | `json_expect` | Important next-token prior | Currently overclaims; `AFTER_KEY` not real | Keep, redesign into real key/colon/value/comma substates |
| 6 | `url_phase` | Useful host/path/query/fragment signal | Needs better reset logic | Keep, redesign boundary handling |
| 7 | `markdown_phase` | Best coarse markdown switch | Still brittle and misfires on some HTML-like cues | Keep, redesign line/block persistence |
| 8 | `code_phase` | Useful coarse code-region state | Overlaps with `mode`, but still valuable | Keep |
| 9 | `quote_mode` | Important delimiter identity | Not a real nested quote stack | Keep, but eventually make nesting-aware |
| 10 | `scope_state` | Helpful protected-span summary | Composite and misleading in nested cases | Keep for now, but mark as replacement candidate if nested scope stack arrives |
| 11 | `quote_len_bucket` | Mildly useful progress signal inside literals | Low priority, somewhat expendable | Keep for now |
| 12 | `prose_state` | Useful shallow NL order cue | Heuristic, but distinct | Keep |
| 13 | `sentence_len_bucket` | Useful sentence age/position cue | Not redundant | Keep |
| 14 | `clause_len_bucket` | Clause-local age cue | Partly redundant with sentence/prose stack | Keep short-term; merge candidate if slots get tight |
| 15 | `bracket_depth` | Strong near-distance nesting cue | Distinct from full stack type | Keep |
| 16 | `same_class_run_bucket` | Tokenization/run-length regularity | Weak grammar signal | Simplify or drop if slot pressure appears |
| 17 | `punct_state` | Last punctuation class memory | Overlaps with `serial_edge_state` | Merge candidate |
| 18 | `stack_top_type` | One of the strongest serial structure features | Clearly justified | Keep |
| 19 | `stack_depth_bucket` | Strong generic nesting depth | Clearly justified | Keep |
| 20 | `stack_parent_type` | Valuable parent-context cue | Distinct from top/depth | Keep |
| 21 | `html_top_tag_class` | Semantic class for active HTML region | Useful and distinct | Keep |
| 22 | `html_tag_depth_bucket` | HTML-specific depth | Partly redundant with generic stack depth | Keep for now; merge candidate |
| 23 | `json_container_top` | Object vs array bias can help | Not a true stack today; partly redundant | Keep only if upgraded to true stack-backed value; otherwise remove later |
| 24 | `markdown_heading_level` | High-value local markdown cue | Current implementation should store actual level, not cumulative artifact | Keep, redesign |
| 25 | `markdown_list_depth` | Valuable nested list cue | Distinct enough | Keep |
| 26 | `markdown_table_col_bucket` | Could help table-local next-token bias | Currently overclaimed pipe-count proxy | Simplify or replace with `in_table`/coarse column index |
| 27 | `markdown_fence_state` | Strong local code-block cue | Clearly justified | Keep |
| 28 | `indent_delta` | Strong short-range transition cue | Useful even if derivable | Keep |
| 29 | `blank_line_run_bucket` | Some discourse-segmentation value | Weak and often flat | Simplify to boolean or drop if needed |
| 30 | `code_after_dot` | Captures local member-chain bias | Too weak as a one-token latch | Replace with richer `member_chain_state` |
| 31 | `code_assignment_side` | Useful left/right boundary cue | Too code-specific for current needs | Generalize into cross-domain `key_value_side` |
| 32 | `code_call_arg_bucket` | Valuable for call/arg local order | Good enough, but needs better call detection | Keep |
| 33 | `prose_clause_stack` | Helpful local clause-role cue | More latch than true stack | Keep, but rename or redesign if it stays shallow |
| 34 | `serial_edge_state` | Best current local transition feature | Too local on delimiter token only | Keep, but add post-delimiter follow-through |
| 35 | `abbrev_state` | Useful sentence-boundary guard | Could be simpler | Keep, maybe collapse to smaller chain model later |

## Merge / Removal Candidates

High-confidence merge or simplification candidates:

- `punct_state` -> merge into a stronger transition-focused feature family built around `serial_edge_state`.
- `code_after_dot` -> replace with `member_chain_state` rather than keeping a one-token latch.
- `code_assignment_side` -> replace with cross-domain `key_value_side`.
- `same_class_run_bucket` -> shrink to a binary repeat flag or drop.
- `blank_line_run_bucket` -> shrink to boolean `blank_line_break` unless longer paragraph-gap memory proves useful.
- `markdown_table_col_bucket` -> replace with coarser `in_table` or `table_col_class`.

Conditional removals if feature budget gets tight:

- `scope_state` if a nested scope stack is implemented and can be derived cheaply.
- `html_tag_depth_bucket` if generic stack depth proves sufficient.
- `json_container_top` unless it becomes truly stack-backed.
- `clause_len_bucket` if `sentence_len_bucket + prose_clause_stack` already cover the signal.

## New Feature Candidates

Prioritized additions:

1. `followup_edge_state`
   Carries a delimiter transition one token past the delimiter itself.
   This is the single highest-value next feature for short-range serial logic.

2. `key_value_side`
   Generalizes assignment-side across code, JSON, HTML attrs, and metadata lines.

3. `member_chain_state`
   Replaces `code_after_dot` with richer local order around `.`, `(`, `)`, `[`, `]`.

4. `nested_scope_top`
   Only if we decide to replace `scope_state` with a true stack-backed scope cue.

5. `abbrev_guard_small`
   Optional simplified replacement for the current multi-state abbreviation tracker if we need to trim dimensionality.

## Recommended Next Patch

Recommended order:

1. Keep the current 36-feature bank for now.
2. Implement `followup_edge_state`.
3. Replace `code_assignment_side` with cross-domain `key_value_side`.
4. Replace `code_after_dot` with `member_chain_state`.
5. Revisit `punct_state`, `same_class_run_bucket`, `blank_line_run_bucket`, and `markdown_table_col_bucket` only after the above changes land.

Net conclusion:

- The current bank is not obviously too large.
- The main problem is not “too many features,” but “a few weak summaries and a few overclaimed names.”
- The next gains are most likely to come from improving local transition features, not from adding more broad global grammar summaries.
