from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ModuleNotFoundError:
    triton = None
    tl = None
    HAS_TRITON = False


def _resolve_score_scale(q: torch.Tensor, score_scale: float | None) -> float:
    return float(score_scale) if score_scale is not None else 1.0 / math.sqrt(q.size(1))


def fused_ngram_attn_reference(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    slot_bias: torch.Tensor | None = None,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Reference implementation for candidate-level softmax attention.

    This matches the core `attn_lite` math in `train_gpt.py`:

        scores_i = (q · k_i) / sqrt(attn_dim) + bias_i
        alpha = softmax(scores over candidate slots)
        out = sum_i alpha_i * v_i

    Shapes:
        q:            [N, A]
        candidate_ids:[N, K]
        key_weight:   [V, A]
        value_weight: [V, D]
        slot_bias:    [K] optional
        output:       [N, D]
    """
    if q.ndim != 2:
        raise ValueError(f"q must have shape [N, A], got {tuple(q.shape)}")
    if candidate_ids.ndim != 2:
        raise ValueError(f"candidate_ids must have shape [N, K], got {tuple(candidate_ids.shape)}")
    if key_weight.ndim != 2:
        raise ValueError(f"key_weight must have shape [V, A], got {tuple(key_weight.shape)}")
    if value_weight.ndim != 2:
        raise ValueError(f"value_weight must have shape [V, D], got {tuple(value_weight.shape)}")
    if q.size(0) != candidate_ids.size(0):
        raise ValueError("q and candidate_ids must agree on token dimension")
    if q.size(1) != key_weight.size(1):
        raise ValueError("q and key_weight must agree on attention dimension")
    if key_weight.size(0) != value_weight.size(0):
        raise ValueError("key_weight and value_weight must agree on embedding row count")
    if slot_bias is not None and slot_bias.shape != (candidate_ids.size(1),):
        raise ValueError("slot_bias must have shape [K]")

    scale = _resolve_score_scale(q, score_scale)
    keys = F.embedding(candidate_ids, key_weight)
    values = F.embedding(candidate_ids, value_weight)
    scores = (q.unsqueeze(1) * keys).sum(dim=-1) * scale
    if slot_bias is not None:
        scores = scores + slot_bias.to(dtype=scores.dtype, device=scores.device)[None, :]
    weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
    return (weights * values).sum(dim=-2)


if HAS_TRITON:
    @triton.jit
    def _fused_ngram_attn_kernel(
        q_ptr,
        cand_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        num_tokens,
        value_dim,
        score_scale,
        stride_qn,
        stride_qa,
        stride_cn,
        stride_ck,
        stride_kr,
        stride_ka,
        stride_vr,
        stride_vd,
        stride_on,
        stride_od,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_ATTN_DIM: tl.constexpr,
        BLOCK_VALUE_DIM: tl.constexpr,
        NUM_CANDIDATES: tl.constexpr,
        ATTN_DIM: tl.constexpr,
    ):
        pid_tok = tl.program_id(0)
        pid_val = tl.program_id(1)

        tok_offsets = pid_tok * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        val_offsets = pid_val * BLOCK_VALUE_DIM + tl.arange(0, BLOCK_VALUE_DIM)

        tok_mask = tok_offsets < num_tokens
        val_mask = val_offsets < value_dim
        out_mask = tok_mask[:, None] & val_mask[None, :]

        running_max = tl.full((BLOCK_TOKENS,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_TOKENS,), dtype=tl.float32)
        running_out = tl.zeros((BLOCK_TOKENS, BLOCK_VALUE_DIM), dtype=tl.float32)

        for cand_idx in tl.static_range(NUM_CANDIDATES):
            cand_ptrs = cand_ptr + tok_offsets * stride_cn + cand_idx * stride_ck
            cand_ids = tl.load(cand_ptrs, mask=tok_mask, other=0).to(tl.int32)

            score = tl.zeros((BLOCK_TOKENS,), dtype=tl.float32)
            for attn_start in tl.static_range(0, ATTN_DIM, BLOCK_ATTN_DIM):
                attn_offsets = attn_start + tl.arange(0, BLOCK_ATTN_DIM)
                attn_mask = attn_offsets < ATTN_DIM
                q_ptrs = q_ptr + tok_offsets[:, None] * stride_qn + attn_offsets[None, :] * stride_qa
                q_tile = tl.load(q_ptrs, mask=tok_mask[:, None] & attn_mask[None, :], other=0.0).to(tl.float32)
                k_ptrs = key_ptr + cand_ids[:, None] * stride_kr + attn_offsets[None, :] * stride_ka
                k_tile = tl.load(k_ptrs, mask=tok_mask[:, None] & attn_mask[None, :], other=0.0).to(tl.float32)
                score += tl.sum(q_tile * k_tile, axis=1)

            bias = tl.load(bias_ptr + cand_idx).to(tl.float32)
            score = score * score_scale + bias

            v_ptrs = value_ptr + cand_ids[:, None] * stride_vr + val_offsets[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=out_mask, other=0.0).to(tl.float32)

            new_max = tl.maximum(running_max, score)
            old_scale = tl.exp(running_max - new_max)
            cur_scale = tl.exp(score - new_max)

            running_sum = running_sum * old_scale + cur_scale
            running_out = running_out * old_scale[:, None] + cur_scale[:, None] * v_tile
            running_max = new_max

        out = running_out / tl.maximum(running_sum[:, None], 1e-9)
        out_ptrs = out_ptr + tok_offsets[:, None] * stride_on + val_offsets[None, :] * stride_od
        tl.store(out_ptrs, out, mask=out_mask)


def _fused_ngram_attn_triton(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    slot_bias: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in this environment")
    if not q.is_cuda or not candidate_ids.is_cuda or not key_weight.is_cuda or not value_weight.is_cuda or not slot_bias.is_cuda:
        raise ValueError("Triton path requires CUDA tensors")
    if key_weight.size(0) != value_weight.size(0):
        raise ValueError("key_weight and value_weight must agree on row count")
    if q.size(1) != key_weight.size(1):
        raise ValueError("q and key_weight must agree on attention dimension")

    num_tokens = q.size(0)
    attn_dim = q.size(1)
    value_dim = value_weight.size(1)
    num_candidates = candidate_ids.size(1)
    out = torch.empty((num_tokens, value_dim), device=q.device, dtype=value_weight.dtype)

    grid = (
        triton.cdiv(num_tokens, 8),
        triton.cdiv(value_dim, 64),
    )
    _fused_ngram_attn_kernel[grid](
        q,
        candidate_ids,
        key_weight,
        value_weight,
        slot_bias,
        out,
        num_tokens,
        value_dim,
        score_scale,
        q.stride(0),
        q.stride(1),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        key_weight.stride(0),
        key_weight.stride(1),
        value_weight.stride(0),
        value_weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_TOKENS=8,
        BLOCK_ATTN_DIM=32,
        BLOCK_VALUE_DIM=64,
        NUM_CANDIDATES=num_candidates,
        ATTN_DIM=attn_dim,
    )
    return out


class FusedNgramAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, candidate_ids, key_weight, value_weight, slot_bias, score_scale):
        ctx.has_slot_bias = slot_bias is not None
        ctx.score_scale = float(score_scale)
        ctx.save_for_backward(
            q,
            candidate_ids,
            key_weight,
            value_weight,
            slot_bias if slot_bias is not None else q.new_zeros(0),
        )
        if HAS_TRITON and q.is_cuda:
            if slot_bias is None:
                slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
            return _fused_ngram_attn_triton(q, candidate_ids, key_weight, value_weight, slot_bias, ctx.score_scale)
        return fused_ngram_attn_reference(
            q,
            candidate_ids,
            key_weight,
            value_weight,
            slot_bias=slot_bias,
            score_scale=ctx.score_scale,
        )

    @staticmethod
    def backward(ctx, grad_out):
        q, candidate_ids, key_weight, value_weight, slot_bias_saved = ctx.saved_tensors
        slot_bias = slot_bias_saved if ctx.has_slot_bias else None

        q_ref = q.detach().requires_grad_(True)
        key_ref = key_weight.detach().requires_grad_(True)
        value_ref = value_weight.detach().requires_grad_(True)
        bias_ref = slot_bias.detach().requires_grad_(True) if slot_bias is not None else None

        with torch.enable_grad():
            out = fused_ngram_attn_reference(
                q_ref,
                candidate_ids,
                key_ref,
                value_ref,
                slot_bias=bias_ref,
                score_scale=ctx.score_scale,
            )

        grad_inputs = (q_ref, key_ref, value_ref, bias_ref) if bias_ref is not None else (q_ref, key_ref, value_ref)
        grads = torch.autograd.grad(out, grad_inputs, grad_out, allow_unused=False)
        if bias_ref is not None:
            grad_q, grad_key, grad_value, grad_bias = grads
        else:
            grad_q, grad_key, grad_value = grads
            grad_bias = None
        return grad_q, None, grad_key, grad_value, grad_bias, None


def fused_ngram_attn(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    slot_bias: torch.Tensor | None = None,
    score_scale: float | None = None,
    impl: str = "auto",
) -> torch.Tensor:
    if impl not in {"auto", "reference", "triton"}:
        raise ValueError(f"Unsupported impl: {impl}")
    resolved_scale = _resolve_score_scale(q, score_scale)
    if impl == "reference":
        return fused_ngram_attn_reference(
            q,
            candidate_ids,
            key_weight,
            value_weight,
            slot_bias=slot_bias,
            score_scale=resolved_scale,
        )
    if impl == "triton":
        if slot_bias is None:
            slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
        return FusedNgramAttnFunction.apply(q, candidate_ids, key_weight, value_weight, slot_bias, resolved_scale)
    if HAS_TRITON and q.is_cuda:
        if slot_bias is None:
            slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
        return FusedNgramAttnFunction.apply(q, candidate_ids, key_weight, value_weight, slot_bias, resolved_scale)
    return fused_ngram_attn_reference(
        q,
        candidate_ids,
        key_weight,
        value_weight,
        slot_bias=slot_bias,
        score_scale=resolved_scale,
    )
