"""
Correctness tests for fused_ngram_single_io.py

Tests:
  1. Forward: Triton vs reference (fp16 & bf16)
  2. Backward: Triton+ref-backward vs pure-reference autograd (grad_q, grad_key, grad_value, grad_bias)
  3. Edge cases: K=1, large K, ATTN_DIM not multiple of BLOCK_ATTN_DIM, VALUE_DIM not multiple of BLOCK_VALUE_DIM
  4. Numerical: gradcheck via torch.autograd.gradcheck on the reference path
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

from fused_ngram_single_io import (
    fused_ngram_attn_reference,
    fused_ngram_attn,
    FusedNgramAttnFunction,
    HAS_TRITON,
)


def _make_inputs(
    N: int,
    K: int,
    A: int,
    D: int,
    V: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    with_bias: bool = True,
    seed: int = 42,
):
    torch.manual_seed(seed)
    q = torch.randn(N, A, device=device, dtype=dtype)
    candidate_ids = torch.randint(0, V, (N, K), device=device, dtype=torch.int32)
    key_weight = torch.randn(V, A, device=device, dtype=dtype)
    value_weight = torch.randn(V, D, device=device, dtype=dtype)
    slot_bias = torch.randn(K, device=device, dtype=torch.float32) * 0.1 if with_bias else None
    return q, candidate_ids, key_weight, value_weight, slot_bias


def test_forward_match(
    N: int = 256,
    K: int = 6,
    A: int = 32,
    D: int = 96,
    V: int = 8192,
    dtype: torch.dtype = torch.float16,
):
    """Check Triton forward matches reference forward."""
    q, cand, kw, vw, bias = _make_inputs(N, K, A, D, V, dtype=dtype)

    ref_out = fused_ngram_attn_reference(q, cand, kw, vw, slot_bias=bias)
    triton_out = fused_ngram_attn(q, cand, kw, vw, slot_bias=bias, impl="triton")

    # Triton runs in fp32 internally, so tolerance is relative to dtype
    if dtype == torch.float16:
        atol, rtol = 1e-2, 5e-3
    else:  # bf16
        atol, rtol = 2e-2, 1e-2

    max_diff = (ref_out.float() - triton_out.float()).abs().max().item()
    mean_diff = (ref_out.float() - triton_out.float()).abs().mean().item()
    ref_norm = ref_out.float().norm().item()

    passed = torch.allclose(ref_out.float(), triton_out.float(), atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] forward {dtype} N={N} K={K} A={A} D={D}: "
        f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} ref_norm={ref_norm:.2f}"
    )
    return passed


def test_backward_match(
    N: int = 128,
    K: int = 6,
    A: int = 32,
    D: int = 96,
    V: int = 4096,
    dtype: torch.dtype = torch.float16,
):
    """Check Triton forward + reference backward gives same grads as pure reference."""
    q, cand, kw, vw, bias = _make_inputs(N, K, A, D, V, dtype=dtype)

    # -- Pure reference path --
    q_ref = q.clone().detach().requires_grad_(True)
    kw_ref = kw.clone().detach().requires_grad_(True)
    vw_ref = vw.clone().detach().requires_grad_(True)
    bias_ref = bias.clone().detach().requires_grad_(True) if bias is not None else None
    out_ref = fused_ngram_attn_reference(q_ref, cand, kw_ref, vw_ref, slot_bias=bias_ref)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # -- Triton forward + reference backward path (via FusedNgramAttnFunction) --
    q_tri = q.clone().detach().requires_grad_(True)
    kw_tri = kw.clone().detach().requires_grad_(True)
    vw_tri = vw.clone().detach().requires_grad_(True)
    bias_tri = bias.clone().detach().requires_grad_(True) if bias is not None else None
    scale = 1.0 / (A ** 0.5)
    out_tri = FusedNgramAttnFunction.apply(
        q_tri, cand, kw_tri, vw_tri,
        bias_tri if bias_tri is not None else torch.zeros(K, device=q.device, dtype=q.dtype),
        scale,
    )
    out_tri.backward(grad_out)

    if dtype == torch.float16:
        atol, rtol = 5e-2, 1e-2
    else:
        atol, rtol = 1e-1, 2e-2

    results = []
    for name, g_ref, g_tri in [
        ("grad_q", q_ref.grad, q_tri.grad),
        ("grad_key", kw_ref.grad, kw_tri.grad),
        ("grad_value", vw_ref.grad, vw_tri.grad),
    ]:
        max_diff = (g_ref.float() - g_tri.float()).abs().max().item()
        passed = torch.allclose(g_ref.float(), g_tri.float(), atol=atol, rtol=rtol)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] backward {name} {dtype}: max_diff={max_diff:.2e}")
        results.append(passed)

    if bias_ref is not None and bias_tri is not None:
        # bias grad: the custom backward gives grad for bias_tri
        # but in FusedNgramAttnFunction, when has_slot_bias=True it should return grad_bias
        # Note: bias_tri was passed directly, so it should have grad
        max_diff = (bias_ref.grad.float() - bias_tri.grad.float()).abs().max().item()
        passed = torch.allclose(bias_ref.grad.float(), bias_tri.grad.float(), atol=atol, rtol=rtol)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] backward grad_bias {dtype}: max_diff={max_diff:.2e}")
        results.append(passed)

    return all(results)


def test_edge_cases():
    """Test non-standard shapes that might trip up Triton tiling."""
    results = []

    # K=1: single candidate (degenerate softmax)
    results.append(test_forward_match(N=64, K=1, A=32, D=96, V=1024))

    # K=10: more candidates than typical
    results.append(test_forward_match(N=64, K=10, A=32, D=96, V=2048))

    # A=17: attn_dim not multiple of BLOCK_ATTN_DIM=32
    results.append(test_forward_match(N=64, K=6, A=17, D=96, V=2048))

    # D=100: value_dim not multiple of BLOCK_VALUE_DIM=64
    results.append(test_forward_match(N=64, K=6, A=32, D=100, V=2048))

    # N=3: fewer tokens than BLOCK_TOKENS=8
    results.append(test_forward_match(N=3, K=6, A=32, D=96, V=2048))

    # Large: typical training shape
    results.append(test_forward_match(N=2048, K=6, A=32, D=96, V=8192))

    return all(results)


def test_no_bias():
    """Test path without slot_bias."""
    q, cand, kw, vw, _ = _make_inputs(128, 6, 32, 96, 4096)

    ref_out = fused_ngram_attn_reference(q, cand, kw, vw, slot_bias=None)
    # Triton path always gets a zero bias when None
    zero_bias = torch.zeros(6, device=q.device, dtype=q.dtype)
    triton_out = fused_ngram_attn(q, cand, kw, vw, slot_bias=zero_bias, impl="triton")

    max_diff = (ref_out.float() - triton_out.float()).abs().max().item()
    passed = max_diff < 1e-2
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] forward no_bias: max_diff={max_diff:.2e}")
    return passed


def test_gradcheck_reference():
    """Finite-difference gradient check on the reference path (fp64)."""
    N, K, A, D, V = 8, 4, 8, 16, 64
    q, cand, kw, vw, bias = _make_inputs(N, K, A, D, V, dtype=torch.float64, device="cuda")
    q.requires_grad_(True)
    kw.requires_grad_(True)
    vw.requires_grad_(True)
    if bias is not None:
        bias = bias.double().requires_grad_(True)

    def fn(q_, kw_, vw_, bias_):
        return fused_ngram_attn_reference(q_, cand, kw_, vw_, slot_bias=bias_)

    passed = torch.autograd.gradcheck(fn, (q, kw, vw, bias), eps=1e-5, atol=1e-3, rtol=1e-3)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] gradcheck reference (fp64)")
    return passed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    if not HAS_TRITON:
        print("Triton not available, only testing reference path")

    all_passed = True

    print("\n=== Forward: Triton vs Reference ===")
    if HAS_TRITON:
        all_passed &= test_forward_match(dtype=torch.float16)
        all_passed &= test_forward_match(dtype=torch.bfloat16)
    else:
        print("  [SKIP] Triton not available")

    print("\n=== Forward: Edge Cases ===")
    if HAS_TRITON:
        all_passed &= test_edge_cases()
    else:
        print("  [SKIP] Triton not available")

    print("\n=== Forward: No Bias ===")
    if HAS_TRITON:
        all_passed &= test_no_bias()
    else:
        print("  [SKIP] Triton not available")

    print("\n=== Backward: Gradient Match ===")
    if HAS_TRITON:
        all_passed &= test_backward_match(dtype=torch.float16)
        all_passed &= test_backward_match(dtype=torch.bfloat16)
    else:
        print("  [SKIP] Triton not available")

    print("\n=== Gradcheck: Reference (fp64) ===")
    all_passed &= test_gradcheck_reference()

    print(f"\n{'='*40}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
