# M35: Shared Latent Code Attention + MLP

This folder tracks `M35`, a backbone parameter-sharing experiment that tries to
generate both attention and MLP weights from shared latent codes without
introducing an extra dense latent mixing matrix.

## Current Code Baseline

The working `train_gpt.py` in this folder started as a direct copy of the March
25 legal SOTA stack:

- [train_gpt.py](/Users/yichengxia/ML_NN_DA/parameter_golf/parameter-golf-base/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py)

Reason for using this baseline:

- it already stored attention and MLP weights in layer banks
- it already carried the training / export / evaluation machinery we wanted as
  a starting point
- replacing the banks with shared latent-code generators is cleaner than
  starting from a fully unbanked script

Current implementation status:

- `train_gpt.py` now uses shared latent-code generators instead of the old dense
  attention/MLP banks
- `train_gpt_mlx.py` now exists as a separate MLX-only skeleton line, copied
  from the repo baseline and ported to the same shared latent-code idea
- GPTQ / Hessian calibration has been removed from the active path
- export quantizes the latent parameter state directly instead of expanding back
  into dense per-layer weights
- v1 currently requires `NUM_KV_HEADS == NUM_HEADS`
- the MLX line is intentionally simpler than the PyTorch line and is meant for
  basic Apple-silicon experiments, not parity with every PyTorch feature

## Core Parameterization

Let:

- `d` = model width
- `H` = number of attention heads
- `d_h = d / H` = per-head width
- `m` = MLP hidden width
- `r_attn` = attention latent code width
- `r_ffn` = MLP latent code width
- `l` = layer index
- `h` = head index

## Attention Shared Code

For each layer `l` and head `h`, store a code matrix:

- `C_attn^(l,h) ∈ R^{d_h x r_attn}`

For each head `h`, share four decoder matrices across layers:

- `D_Q^(h) ∈ R^{r_attn x d}`
- `D_K^(h) ∈ R^{r_attn x d}`
- `D_V^(h) ∈ R^{r_attn x d}`
- `D_O^(h) ∈ R^{r_attn x d}`

Then define the actual per-head projections as:

- `W_Q^(l,h) = C_attn^(l,h) D_Q^(h) ∈ R^{d_h x d}`
- `W_K^(l,h) = C_attn^(l,h) D_K^(h) ∈ R^{d_h x d}`
- `W_V^(l,h) = C_attn^(l,h) D_V^(h) ∈ R^{d_h x d}`
- `W_O^(l,h)^T = C_attn^(l,h) D_O^(h) ∈ R^{d_h x d}`

Interpretation:

- each column `c_(l,h,j)` of `C_attn^(l,h)` is one latent attention atom
- the same code column generates how that channel reads as `Q / K`, carries
  content as `V`, and writes back through `O`
- the main sharing happens through `C_attn`
- the main reusable decoder structure happens through `D_*`

## MLP Shared Code

For a standard two-matrix FFN with

- `W_up^(l) ∈ R^{d x m}`
- `W_down^(l) ∈ R^{m x d}`

store one code matrix per layer:

- `C_ffn^(l) ∈ R^{m x r_ffn}`

and share two decoder matrices across layers:

- `D_up ∈ R^{r_ffn x d}`
- `D_down ∈ R^{r_ffn x d}`

Then define:

- `W_up^(l) = C_ffn^(l) D_up ∈ R^{m x d}`
- `W_down^(l)^T = C_ffn^(l) D_down ∈ R^{m x d}`

equivalently:

- `W_down^(l) = D_down^T C_ffn^(l)^T ∈ R^{d x m}`

Interpretation:

- each hidden channel gets one latent code column from `C_ffn^(l)`
- that same code decides both how the layer reads into the hidden MLP channel
  and how it writes back to the residual stream
- this is the MLP analogue of the shared `Q / K / V / O` code in attention

If the stack uses a gated FFN such as SwiGLU, the simplest extension is:

- add `D_gate ∈ R^{r_ffn x d}`
- define `W_gate^(l) = C_ffn^(l) D_gate`

The first implementation pass does not need to force gated FFN support, but the
shared-code rule should extend naturally to it.

## Important Constraint

This design intentionally does **not** include an extra dense `M` matrix between
decoder matrices and latent codes.

Reason:

- if `M` is a free dense linear map at the same locality as the code tensor,
  then `D_* M C` just collapses into a reparameterized `C` or `D_*`
- that moves the design back toward generic factorization instead of the
  intended "shared code generates multiple projections" idea

Small structured gates may still be worth testing later, but the first version
should stay `decoder * code` only.

## Why This Is Not Ordinary Low-Rank Factorization

This is **not** mainly trying to make each attention or MLP matrix low-rank.

Instead:

- each head can still be full-rank as long as `r_attn >= d_h`
- each FFN matrix can still reach full input/output rank as long as
  `r_ffn >= d`
- compression comes from reusing decoder structure across `Q / K / V / O`,
  across MLP read/write paths, and across layers
- the bottleneck is the shared latent code space, not a forced low-rank
  factorization on the full dense matrices

## Parameter Budget: Attention

Standard attention stack with `L` layers:

- `4 L d^2`

M35 with head-local decoders shared across layers:

- decoder params: `4 H d r_attn`
- per-layer codes: `L d r_attn`
- total: `d r_attn (4 H + L)`

Compression condition:

- `d r_attn (4 H + L) < 4 L d^2`
- equivalently `r_attn < 4 L d / (4 H + L)`

For the common `11L / d=512 / H=8` case:

- `d_h = 64`
- compression still holds for any `r_attn < 523`

That means we can keep `r_attn >= d_h` and still remain parameter-efficient
once the sharing happens across layers.

## Parameter Budget: MLP

Standard two-matrix FFN stack with `L` layers:

- `2 L d m`

M35 with layer-local FFN codes and decoders shared across layers:

- decoder params: `2 d r_ffn`
- per-layer codes: `L m r_ffn`
- total: `r_ffn (2 d + L m)`

Compression condition:

- `r_ffn (2 d + L m) < 2 L d m`
- equivalently `r_ffn < 2 L d m / (2 d + L m)`

For a common `11L / d=512 / m=1536` setup:

- compression still holds for any `r_ffn < 966`

So the MLP path has even more room than attention because the FFN width gives
us the expansion attention lacks.

## First Implementation Target

- keep the normal attention and MLP compute paths intact
- materialize dense per-head `Q / K / V / O` weights from `D_*` and `C_attn`
  before the usual attention call
- materialize dense `W_up / W_down` from `D_*` and `C_ffn` before the usual
  FFN call
- start with attention decoders shared across layers but **not** across heads
- start with MLP decoders shared across layers and global across FFN layers
- keep attention and MLP latent spaces separate in v1
- first sweep `r_attn` over `{64, 80, 96}`
- first sweep `r_ffn` over `{512, 640, 768}`
- compare against the same training stack with ordinary dense weights

## Main Questions

- does sharing one `C_attn` across `Q / K / V / O` over-couple the attention
  head?
- does sharing one `C_ffn` across `W_up / W_down` over-couple the FFN hidden
  channels?
- is the gain still real after export and compression?
- can we keep throughput reasonable if dense weights are materialized on the fly?
- does cross-layer sharing help enough to justify the extra implementation
  complexity?

This folder now contains the first implementation pass. The next step is to run
cheap smoke tests and see whether the latent materialization overhead is still
compatible with the 10-minute training regime.

## Mac Overnight Runs

On MacBooks, a long MLX job can be interrupted by idle sleep even when the
script itself still has remaining wallclock budget. For overnight runs, launch
the training command through:

```bash
./run_caffeinated.sh python3 train_gpt_mlx.py
```

The wrapper uses `caffeinate -i -m -s` so the system stays awake for the
duration of the training process while still allowing the display to sleep. If
you need custom environment variables, put them before the wrapper command, for
example:

```bash
MAX_WALLCLOCK_SECONDS=32400 TRAIN_LOG_EVERY=20 \
./run_caffeinated.sh python3 train_gpt_mlx.py
```
