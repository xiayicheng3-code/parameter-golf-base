from __future__ import annotations

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


def fused_ngram_softmax_reference(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    slot_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError(f"q must have shape [N, D], got {tuple(q.shape)}")
    if candidate_ids.ndim != 2:
        raise ValueError(f"candidate_ids must have shape [N, K], got {tuple(candidate_ids.shape)}")
    if embed_weight.ndim != 2:
        raise ValueError(f"embed_weight must have shape [V, D], got {tuple(embed_weight.shape)}")
    if q.size(0) != candidate_ids.size(0):
        raise ValueError("q and candidate_ids must agree on token dimension")
    if q.size(1) != embed_weight.size(1):
        raise ValueError("q and embed_weight must agree on feature dimension")
    if slot_bias is not None and slot_bias.shape != (candidate_ids.size(1),):
        raise ValueError("slot_bias must have shape [K]")

    values = F.embedding(candidate_ids, embed_weight)
    logits = q.unsqueeze(1) * values
    if slot_bias is not None:
        logits = logits + slot_bias.to(dtype=logits.dtype, device=logits.device)[None, :, None]
    weights = torch.softmax(logits, dim=1)
    return (weights * values).sum(dim=1)


if HAS_TRITON:
    @triton.jit
    def _fused_ngram_softmax_kernel(
        q_ptr,
        cand_ptr,
        emb_ptr,
        bias_ptr,
        out_ptr,
        num_tokens,
        dim,
        stride_qn,
        stride_qd,
        stride_cn,
        stride_ck,
        stride_ev,
        stride_ed,
        stride_on,
        stride_od,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        NUM_CANDIDATES: tl.constexpr,
    ):
        pid_tok = tl.program_id(0)
        pid_dim = tl.program_id(1)

        tok_offsets = pid_tok * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        dim_offsets = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

        tok_mask = tok_offsets < num_tokens
        dim_mask = dim_offsets < dim
        mask = tok_mask[:, None] & dim_mask[None, :]

        q_ptrs = q_ptr + tok_offsets[:, None] * stride_qn + dim_offsets[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)

        running_max = tl.full((BLOCK_TOKENS, BLOCK_DIM), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_TOKENS, BLOCK_DIM), dtype=tl.float32)
        running_out = tl.zeros((BLOCK_TOKENS, BLOCK_DIM), dtype=tl.float32)

        for k in tl.static_range(NUM_CANDIDATES):
            cand_ptrs = cand_ptr + tok_offsets * stride_cn + k * stride_ck
            cand_ids = tl.load(cand_ptrs, mask=tok_mask, other=0).to(tl.int32)
            emb_ptrs = emb_ptr + cand_ids[:, None] * stride_ev + dim_offsets[None, :] * stride_ed
            v = tl.load(emb_ptrs, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(bias_ptr + k).to(tl.float32)
            logits = q * v + bias

            new_max = tl.maximum(running_max, logits)
            old_scale = tl.exp(running_max - new_max)
            cur_scale = tl.exp(logits - new_max)

            running_sum = running_sum * old_scale + cur_scale
            running_out = running_out * old_scale + cur_scale * v
            running_max = new_max

        out = running_out / running_sum
        out_ptrs = out_ptr + tok_offsets[:, None] * stride_on + dim_offsets[None, :] * stride_od
        tl.store(out_ptrs, out, mask=mask)


def _fused_ngram_softmax_triton(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    slot_bias: torch.Tensor,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in this environment")
    if not q.is_cuda or not candidate_ids.is_cuda or not embed_weight.is_cuda or not slot_bias.is_cuda:
        raise ValueError("Triton path requires CUDA tensors")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"q must be fp16/bf16 for Triton path, got {q.dtype}")

    num_tokens, dim = q.shape
    num_candidates = candidate_ids.size(1)
    out = torch.empty_like(q)

    grid = (
        triton.cdiv(num_tokens, 8),
        triton.cdiv(dim, 64),
    )
    _fused_ngram_softmax_kernel[grid](
        q,
        candidate_ids,
        embed_weight,
        slot_bias,
        out,
        num_tokens,
        dim,
        q.stride(0),
        q.stride(1),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        embed_weight.stride(0),
        embed_weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_TOKENS=8,
        BLOCK_DIM=64,
        NUM_CANDIDATES=num_candidates,
    )
    return out


class FusedNgramSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, candidate_ids, embed_weight, slot_bias):
        ctx.has_slot_bias = slot_bias is not None
        ctx.save_for_backward(q, candidate_ids, embed_weight, slot_bias if slot_bias is not None else q.new_zeros(0))
        if HAS_TRITON and q.is_cuda:
            if slot_bias is None:
                slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
            return _fused_ngram_softmax_triton(q, candidate_ids, embed_weight, slot_bias)
        return fused_ngram_softmax_reference(q, candidate_ids, embed_weight, slot_bias=slot_bias)

    @staticmethod
    def backward(ctx, grad_out):
        q, candidate_ids, embed_weight, slot_bias_saved = ctx.saved_tensors
        slot_bias = slot_bias_saved if ctx.has_slot_bias else None

        q_ref = q.detach().requires_grad_(True)
        embed_ref = embed_weight.detach().requires_grad_(True)
        bias_ref = slot_bias.detach().requires_grad_(True) if slot_bias is not None else None

        with torch.enable_grad():
            out = fused_ngram_softmax_reference(
                q_ref,
                candidate_ids,
                embed_ref,
                slot_bias=bias_ref,
            )
        grads = torch.autograd.grad(
            out,
            (q_ref, embed_ref, bias_ref) if bias_ref is not None else (q_ref, embed_ref),
            grad_out,
            allow_unused=False,
        )

        if bias_ref is not None:
            grad_q, grad_embed, grad_bias = grads
        else:
            grad_q, grad_embed = grads
            grad_bias = None
        return grad_q, None, grad_embed, grad_bias


def fused_ngram_softmax(
    q: torch.Tensor,
    candidate_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    slot_bias: torch.Tensor | None = None,
    impl: str = "auto",
) -> torch.Tensor:
    if impl not in {"auto", "reference", "triton"}:
        raise ValueError(f"Unsupported impl: {impl}")
    if impl == "reference":
        return fused_ngram_softmax_reference(q, candidate_ids, embed_weight, slot_bias=slot_bias)
    if impl == "triton":
        if slot_bias is None:
            slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
        return FusedNgramSoftmaxFunction.apply(q, candidate_ids, embed_weight, slot_bias)
    if HAS_TRITON and q.is_cuda:
        if slot_bias is None:
            slot_bias = torch.zeros(candidate_ids.size(1), device=q.device, dtype=q.dtype)
        return FusedNgramSoftmaxFunction.apply(q, candidate_ids, embed_weight, slot_bias)
    return fused_ngram_softmax_reference(q, candidate_ids, embed_weight, slot_bias=slot_bias)
