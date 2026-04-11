# Transformer State-Only Grammar v1

This run rebuilds the web-grammar idea as a portable transformer input feature module.

## Intent

The state machine is only an input encoder. It does not build a count table, does not produce `bucket_lp`, and does not combine logits as `base_lp + logit_bias`.

The model learns:

```text
logits = transformer(token_context + grammar_state_embedding)
```

## Files

- `grammar_features.py`: portable grammar adapter. It builds token metadata from the SentencePiece vocab, runs a shallow web/prose state machine over `input_ids`, and returns a learned embedding with a scalar gate.
- `train_gpt.py`: a small runnable scaffold for local smoke tests. It exists to test the adapter, not to be the final SOTA host.

## State Features

Static single-token metadata such as token class, lexical role, and shape is kept internal for state transitions, but it is not emitted directly to the Transformer. The adapter emits 32 categorical state features that depend on context across at least one token boundary:

```text
line_state
indent_bucket
mode
html_phase
json_depth
json_expect
url_phase
markdown_phase
code_phase
quote_mode
quote_len_bucket
prose_state
sentence_len_bucket
clause_len_bucket
bracket_depth
same_class_run_bucket
punct_state
stack_top_type
stack_depth_bucket
html_top_tag_class
html_tag_depth_bucket
json_container_top
markdown_heading_level
markdown_list_depth
markdown_table_col_bucket
markdown_fence_state
indent_delta
blank_line_run_bucket
code_after_dot
code_assignment_side
code_call_arg_bucket
prose_clause_stack
```

These are embedded separately, summed, and scaled by a learned gate before being added to the token stream.

## Portability Notes

To graft this into a stronger PR stack:

1. Copy `grammar_features.py` next to that stack's `train_gpt.py`.
2. Build `tables = build_vocab_tables_from_sentencepiece(sp, vocab_size)` after loading the tokenizer.
3. Add `self.grammar_adapter = GrammarStateEmbedding(tables, model_dim, init_scale=0.0)` in the model.
4. Change the embedding line to `x = self.tok_emb(input_ids) + self.grammar_adapter(input_ids)`.
5. Add the adapter parameters to an Adam parameter group.

Use `init_scale=0.0` for a baseline-preserving graft. In standalone smoke tests, a small nonzero `GRAMMAR_INIT_SCALE` can make the adapter learn sooner.

## Local Smoke

Run from this folder:

```bash
DEVICE=cpu \
TRAIN_TOKEN_LIMIT=50000 \
VAL_TOKEN_LIMIT=20000 \
SEQ_LEN=64 \
BATCH_SIZE=8 \
STEPS=5 \
D_MODEL=96 \
N_LAYERS=2 \
N_HEADS=4 \
/Users/yichengxia/ML_NN_DA/.venv/bin/python train_gpt.py
```

For a token-only control, add `USE_GRAMMAR=0`.

## Smoke Results

These are only scaffold checks, not quality claims:

| Run | Command delta | `val_loss` | `val_bpb` | Total bytes |
| --- | --- | ---: | ---: | ---: |
| `state_only_32state_stackfix` | 32-state cross-token adapter | `6.921273` | `4.028574` | `300622` |
| `state_only_32state_stackfix_token_control` | `USE_GRAMMAR=0` | `6.931731` | `4.034661` | `274451` |

Both used `DEVICE=cpu TRAIN_TOKEN_LIMIT=50000 VAL_TOKEN_LIMIT=20000 SEQ_LEN=64 BATCH_SIZE=4 STEPS=3 D_MODEL=64 N_LAYERS=1 N_HEADS=4 EVAL_EVERY=0`.

The current adapter is intentionally simple and stateful. It is easy to graft, but the Python loop over sequence positions is the first performance risk to address before putting it into a serious H100 run.
