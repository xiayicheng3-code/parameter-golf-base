# Operator Analysis: Control Baseline vs Experimental All-In

This note explains why the new control baseline is dramatically faster than the earlier
experimental runs in the same folder.

## Observed Step Time Gap

Experimental runs from this folder:

- `log0325.txt`: `~1160 ms/step`, `45.1M` params, 17 effective layers, full all-in path
- `log0324_2.txt`: `~1515 ms/step`, `61.1M` params, 21 effective layers, dual-loop path

Current control baseline:

- user-reported cloud run after the recent fixes: `~280 ms/step`

So the control baseline is roughly:

- `4.1x` faster than the 17-layer all-in run
- `5.4x` faster than the 21-layer dual-loop run

This is too large to be explained by parameter count alone. The main difference is operator
selection and operator shape.

## Main Causes

### 1. Attention path changed from explicit masked local attention to causal SDPA fast path

This is the single biggest reason.

Earlier experimental runs used:

- native sliding-window attention
- explicit `attn_mask`
- chunked evaluation over the sequence
- `math=True` fallback because masked sliding under `torch.compile` could not reliably stay on flash

That path creates large intermediate tensors and launches many more kernels. It also prevents the
clean causal flash-style fast path.

The control baseline now hits this fast path in `SlidingCausalSelfAttention.forward(...)`:

- no shifted attention
- no local sliding restriction during training
- no explicit attention mask
- `is_causal=True`

That means the backend can execute a much more efficient fused causal SDPA kernel instead of
materializing masked chunk-by-chunk attention.

Practical effect:

- fewer kernel launches
- much less temporary memory traffic
- no huge explicit mask tensors
- far better kernel fusion and tiling

This was also the reason for the earlier `2048`-length OOM: the old control run accidentally fell
back to the masked path and tried to allocate a giant dense attention workspace.

### 2. `GRAD_ACCUM_STEPS` went from implicit 8 to explicit 1

Previously this script hardcoded:

- `grad_accum_steps = 8 // world_size`

So on a single GPU, one “step” actually meant 8 forward/backward microsteps.

Now the baseline run used:

- `GRAD_ACCUM_STEPS=1`

This alone is a major step-time reduction. Even if the kernel mix had stayed the same, the
measured wallclock per logged step would have dropped sharply.

So the new `~280 ms/step` is not only a better operator path, but also a more honest single-step
measurement.

### 3. The experimental path had many extra small operators around each major GEMM

The all-in model added several per-layer or per-pass operator fragments:

- full-history `M07` aggregation
- loop-path effective weight materialization
- per-pass LoRA fusion
- loop-path `LayerRoPE` column rotation
- shifted-attention key shifting
- weight-noise injection
- optional deep XSA projection logic

Each individual piece is not huge, but together they fragment the forward pass:

- more launches
- more dtype/device casts
- more temporary tensors
- less regular compute structure

The control baseline removes almost all of these from the hot path:

- no `M07`
- no loop adapters
- no loop `LayerRoPE`
- no weight noise
- no shifted attention

So the baseline is not just “smaller”; it is much less fragmented.

### 4. The control baseline restored a normal residual-style block, which is operator-cheaper

The earlier experimental branch used a much more exotic block dynamic:

- root plus all-history mixing before each block
- no residual carry-through inside the block
- explicit history storage and reuse

That forces more tensor bookkeeping and less standard dataflow.

The control baseline now uses a more normal residual/U-Net style block:

- residual-style `x + attn + mlp`
- optional `resid_mix` with `x0`
- shallow skip structure

This is closer to the pattern compilers and fused kernels handle well.

### 5. Parameter count did matter, but it was not the dominant factor

Parameter counts:

- control baseline: `26.996M`
- experimental all-in: `45.095M`
- larger dual-loop experiment: `61.099M`

So the baseline does have fewer parameters, and that helps.

But parameter ratio alone does not explain a `4x-5x` speedup:

- `45.1M -> 27.0M` is only about `1.67x` smaller
- `61.1M -> 27.0M` is only about `2.26x` smaller

The rest of the speedup comes from better kernels and fewer side operators.

## Why the old path was especially bad

The earlier all-in design combined several expensive traits at once:

- sliding window implemented with explicit masks
- chunked SDPA
- many loop passes
- loop adapter effective weights
- full-history `M07`
- `grad_accum_steps=8`

That combination is bad for throughput because it attacks all three cost centers at once:

1. Large matmul/attention FLOPs
2. Kernel launch count
3. Memory bandwidth / temporary tensor traffic

The control baseline improved all three simultaneously.

## Approximate contribution breakdown

Not exact, but directionally:

- biggest win: causal SDPA fast path instead of masked sliding attention
- second biggest win: `GRAD_ACCUM_STEPS=1` instead of `8`
- third biggest win: removing loop/M07/LoRA/LayerRoPE side operators from the hot path
- fourth win: lower parameter count and simpler block topology

## What this means for future experiments

The lesson is not merely “the baseline is simpler.”

The stronger lesson is:

- operator choice matters as much as architecture choice
- explicit masks and fragmented per-pass logic can completely hide whether an architecture is
  actually good
- before judging a new backbone idea, we need to make sure it still lands on efficient kernels

In particular, any future attempt to revive the experimental path should try to preserve these two
properties:

- keep the main train-time attention path on a causal fast path whenever possible
- avoid adding many tiny operators around every block unless they clearly repay their cost

## Immediate hypothesis

The earlier poor results were likely a mixture of two problems:

1. the experimental backbone may indeed be harder to optimize
2. but the operator path was also so inefficient that it obscured the backbone comparison

Now that the control baseline runs at `~280 ms/step`, we finally have a fair reference point for
judging whether future complexity is buying enough quality to justify its systems cost.

## Why validation no longer feels compile-bound

The same operator story also explains why the new validation path no longer appears to hang on
compile.

### 1. Normal training validation uses `eval_val`, not the special sliding evaluator

During normal training, the script calls `eval_val(...)`, not `eval_val_sliding(...)`.

That matters because:

- `eval_val(...)` just runs the already-compiled training model on validation batches
- `eval_val_sliding(...)` separately does `torch.compile(base_model.forward_logits, ...)`

So the old “validation compile stall” feeling was often a mixture of:

- a genuinely expensive graph
- plus the first time a separate logits-only graph got compiled
- plus the bad masked attention path making tracing and lowering much worse

### 2. The control baseline now reuses a much cleaner graph

Once the control baseline switched to the causal SDPA fast path, the model graph became much more
compiler-friendly:

- no explicit dense attention mask in the main train/val path
- no chunked masked attention loop for full-context validation
- no M07 history aggregation branch
- no loop-adapter fusion branch
- no loop-path LayerRoPE branch
- no weight-noise branch

This gives `torch.compile` a graph that is:

- smaller
- more regular
- more static in tensor shape behavior
- much closer to standard transformer kernels

That reduces both:

- first-time compile latency
- the chance that validation re-enters a slow lowering path

### 3. We also removed useless `step 0` validation

Previously the script validated immediately at the start of training.

That was bad for user experience because:

- warmup ended
- then validation kicked in
- and any one-time compile cost showed up right there

Now `step 0` validation is disabled, so we no longer force that expensive “train has not even
started yet but now we compile/eval anyway” pause.

This does not change asymptotic eval speed very much, but it dramatically improves perceived
responsiveness.

### 4. The old masked local-attention graph was especially hostile to compilation

The experimental path did validation through a much uglier compute pattern:

- local sliding-window attention
- explicit per-chunk causal masks
- chunk loops in Python
- more side branches around each block

That means the compiler had to deal with:

- more graph nodes
- more temporary tensors
- more shape-sensitive branches
- more backend constraints because masked SDPA could not stay on the clean fast path

So validation felt “compile-bound” partly because the graph really was much nastier.

### 5. In practice, the new val path is no longer doing anything exotic

For the current control baseline, validation is basically:

- run the same compiled model structure
- with the same full-context causal attention pattern
- on validation batches instead of training batches

That is exactly the kind of path compilers cache and reuse well.

## Validation-specific conclusion

The disappearance of validation compile stalls is not a mystery and not a lucky accident.

It follows from four concrete changes:

1. the main attention path now lands on causal SDPA fast kernels
2. the validation graph is much simpler and more standard
3. `step 0` validation was removed
4. we stopped dragging masked sliding-window operator complexity through the main validation path

So this improvement is another strong signal that the old performance problems were primarily
systems-path problems, not just “the model needs more tuning.”
