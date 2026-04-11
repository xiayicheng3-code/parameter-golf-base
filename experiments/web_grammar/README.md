# Web Grammar Experiments

This family explores hand-authored grammar state as a small inductive bias for FineWeb compression.

The current direction is intentionally state-only: grammar code may describe the context, but it must not provide a count-table base distribution or a residual log-probability table. The intended graft point is the token embedding stream:

```python
x = tok_emb(input_ids)
x = x + grammar_adapter(input_ids)
```

This keeps the grammar module portable enough to plug into stronger transformer stacks later.
