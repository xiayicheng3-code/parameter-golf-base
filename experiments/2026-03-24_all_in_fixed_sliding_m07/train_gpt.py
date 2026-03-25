from __future__ import annotations
import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    try:
        from flash_attn import flash_attn_func as flash_attn_3_func
    except ImportError:
        flash_attn_3_func = None
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    sliding_window_size = int(os.environ.get("SLIDING_WINDOW_SIZE", 512))
    sliding_chunk_size = int(os.environ.get("SLIDING_CHUNK_SIZE", 0))
    shifted_attention_layers = int(os.environ.get("SHIFTED_ATTENTION_LAYERS", 3))
    num_prelude_layers = int(os.environ.get("NUM_PRELUDE_LAYERS", 2))
    num_loop_layers = int(os.environ.get("NUM_LOOP_LAYERS", 3))
    loop_repeats = int(os.environ.get("LOOP_REPEATS", 2))
    num_epilogue_layers = int(os.environ.get("NUM_EPILOGUE_LAYERS", 3))
    lora_rank = int(os.environ.get("LORA_RANK", 8))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))  # tighter: collect more recent checkpoints
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))  # XSA on last 4 layers (0 = disabled)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.0))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    init_seed = int(os.environ.get("INIT_SEED", os.environ.get("SEED", "1337")))
    init_impl = os.environ.get("INIT_IMPL", "stateless_ortho_v1")
    export_mode = os.environ.get("EXPORT_MODE", "delta_hybrid")
    compression_schemes = tuple(
        part.strip()
        for part in os.environ.get(
            "COMPRESSION_SCHEMES",
            "delta_attn_int8_mlp_int6,delta_attn_int6_mlp_int6,delta_attn_int6_mlp_int4,raw_gptq,raw_int_mixed",
        ).split(",")
        if part.strip()
    )
    delta_int6_cats = tuple(
        part.strip()
        for part in os.environ.get("DELTA_INT6_CATS", "mlp,attn").split(",")
        if part.strip()
    )
    delta_fp16_name_patterns = tuple(
        part.strip()
        for part in os.environ.get("DELTA_FP16_NAME_PATTERNS", "").split(",")
        if part.strip()
    )


def _seed_from_parts(base_seed: int, *parts: str) -> int:
    seed = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        for ch in part.encode("utf-8"):
            seed = ((seed ^ ch) * 16777619) & 0xFFFFFFFF
    return seed


def stateless_randn(shape: tuple[int, ...], base_seed: int, *parts: str) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_from_parts(base_seed, *parts))
    return torch.randn(shape, generator=gen, dtype=torch.float32)


def stateless_normal_(tensor: Tensor, base_seed: int, name: str, std: float) -> None:
    with torch.no_grad():
        tensor.copy_(stateless_randn(tuple(tensor.shape), base_seed, name, "normal").to(dtype=tensor.dtype) * std)


def stateless_orthogonal_(tensor: Tensor, base_seed: int, name: str, gain: float = 1.0) -> None:
    rows, cols = tensor.shape
    q = stateless_randn((rows, cols), base_seed, name, "orthogonal")
    transposed = rows < cols
    if transposed:
        q = q.t()
    q, r = torch.linalg.qr(q, mode="reduced")
    ph = torch.diag(r).sign()
    q *= torch.where(ph == 0, torch.ones_like(ph), ph).unsqueeze(0)
    if transposed:
        q = q.t()
    with torch.no_grad():
        tensor.copy_(q.to(dtype=tensor.dtype) * gain)
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,q_gain,smear,dtg_gate,ve_layer_scales,ve_shared.scale,hyper_conn_weights,layer_rope_theta",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_float_tensor_nbit(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    if bits >= 8:
        return quantize_float_tensor(t)
    qmax = (1 << (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale


def pack_lowbit_tensor(q: Tensor, bits: int) -> Tensor:
    if bits not in {4, 5, 6}:
        raise ValueError(f"Unsupported lowbit pack bits={bits}")
    group_size = math.lcm(8, bits) // bits
    packed_bytes = (group_size * bits) // 8
    qmax = (1 << (bits - 1)) - 1
    vals = (q.detach().to("cpu", dtype=torch.int16).reshape(-1).numpy().astype(np.int16, copy=False) + qmax).astype(np.uint64)
    pad = (-vals.size) % group_size
    if pad:
        vals = np.pad(vals, (0, pad))
    groups = vals.reshape(-1, group_size)
    shifts = np.arange(group_size, dtype=np.uint64) * bits
    packed = np.bitwise_or.reduce(groups << shifts[None, :], axis=1)
    byte_shifts = np.arange(packed_bytes, dtype=np.uint64) * 8
    out = ((packed[:, None] >> byte_shifts[None, :]) & 0xFF).astype(np.uint8).reshape(-1)
    return torch.from_numpy(out.copy())


def unpack_lowbit_tensor(packed: Tensor, bits: int, numel: int) -> Tensor:
    if bits not in {4, 5, 6}:
        raise ValueError(f"Unsupported lowbit unpack bits={bits}")
    group_size = math.lcm(8, bits) // bits
    packed_bytes = (group_size * bits) // 8
    qmax = (1 << (bits - 1)) - 1
    raw = packed.detach().to("cpu", dtype=torch.uint8).reshape(-1).numpy().astype(np.uint64, copy=False)
    if raw.size % packed_bytes != 0:
        raise ValueError(f"Packed tensor byte count {raw.size} is not divisible by {packed_bytes} for {bits}-bit unpack")
    groups = raw.reshape(-1, packed_bytes)
    byte_shifts = np.arange(packed_bytes, dtype=np.uint64) * 8
    merged = np.bitwise_or.reduce(groups << byte_shifts[None, :], axis=1)
    shifts = np.arange(group_size, dtype=np.uint64) * bits
    mask = (1 << bits) - 1
    vals = ((merged[:, None] >> shifts[None, :]) & mask).reshape(-1)[:numel].astype(np.int16)
    vals -= qmax
    return torch.from_numpy(vals.astype(np.int8, copy=False).copy())


def pack_unsigned_lowbit_tensor(codes: Tensor, bits: int) -> Tensor:
    group_size = math.lcm(8, bits) // bits
    packed_bytes = (group_size * bits) // 8
    vals = codes.detach().to("cpu", dtype=torch.uint8).reshape(-1).numpy().astype(np.uint64, copy=False)
    pad = (-vals.size) % group_size
    if pad:
        vals = np.pad(vals, (0, pad))
    groups = vals.reshape(-1, group_size)
    shifts = np.arange(group_size, dtype=np.uint64) * bits
    packed = np.bitwise_or.reduce(groups << shifts[None, :], axis=1)
    byte_shifts = np.arange(packed_bytes, dtype=np.uint64) * 8
    out = ((packed[:, None] >> byte_shifts[None, :]) & 0xFF).astype(np.uint8).reshape(-1)
    return torch.from_numpy(out.copy())


def unpack_unsigned_lowbit_tensor(packed: Tensor, bits: int, numel: int) -> Tensor:
    group_size = math.lcm(8, bits) // bits
    packed_bytes = (group_size * bits) // 8
    raw = packed.detach().to("cpu", dtype=torch.uint8).reshape(-1).numpy().astype(np.uint64, copy=False)
    if raw.size % packed_bytes != 0:
        raise ValueError(f"Packed tensor byte count {raw.size} is not divisible by {packed_bytes} for {bits}-bit unpack")
    groups = raw.reshape(-1, packed_bytes)
    byte_shifts = np.arange(packed_bytes, dtype=np.uint64) * 8
    merged = np.bitwise_or.reduce(groups << byte_shifts[None, :], axis=1)
    shifts = np.arange(group_size, dtype=np.uint64) * bits
    mask = (1 << bits) - 1
    vals = ((merged[:, None] >> shifts[None, :]) & mask).reshape(-1)[:numel].astype(np.uint8)
    return torch.from_numpy(vals.copy())


def quantize_minifloat_tensor(t: Tensor, exp_bits: int, mant_bits: int) -> Tensor:
    total_bits = 1 + exp_bits + mant_bits
    if total_bits not in {6, 8}:
        raise ValueError(f"Unsupported minifloat bitwidth {total_bits}")
    t32 = t.detach().to("cpu", dtype=torch.float32)
    sign = (t32 < 0).to(torch.uint8)
    ax = t32.abs()
    bias = (1 << (exp_bits - 1)) - 1
    max_exp_field = (1 << exp_bits) - 2
    normal_mask = ax > 0
    safe_ax = torch.where(normal_mask, ax, torch.ones_like(ax))
    exp_unbiased = torch.floor(torch.log2(safe_ax)).to(torch.int32)
    biased = exp_unbiased + bias
    valid = normal_mask & (biased > 0)
    biased = torch.clamp(biased, 0, max_exp_field)
    base = torch.pow(torch.tensor(2.0, dtype=torch.float32), (biased - bias).to(torch.float32))
    frac = torch.clamp(ax / base - 1.0, 0.0, 1.0)
    mant = torch.round(frac * (1 << mant_bits)).to(torch.int32)
    carry = mant == (1 << mant_bits)
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    biased = torch.where(carry, torch.clamp(biased + 1, max=max_exp_field), biased)
    codes = ((sign.to(torch.int32) << (exp_bits + mant_bits)) | (biased << mant_bits) | mant).to(torch.uint8)
    codes = torch.where(valid, codes, torch.zeros_like(codes))
    return codes.contiguous()


def dequantize_minifloat_tensor(codes: Tensor, shape: list[int], exp_bits: int, mant_bits: int, dtype_name: str) -> Tensor:
    codes = codes.detach().to("cpu", dtype=torch.uint8).reshape(shape)
    sign = ((codes >> (exp_bits + mant_bits)) & 0x1).to(torch.float32)
    exp_field = ((codes >> mant_bits) & ((1 << exp_bits) - 1)).to(torch.int32)
    mant = (codes & ((1 << mant_bits) - 1)).to(torch.float32)
    bias = (1 << (exp_bits - 1)) - 1
    value = (1.0 + mant / float(1 << mant_bits)) * torch.pow(
        torch.tensor(2.0, dtype=torch.float32),
        (exp_field - bias).to(torch.float32),
    )
    value = torch.where(exp_field > 0, value, torch.zeros_like(value))
    value = torch.where(sign > 0, -value, value)
    return value.to(getattr(torch, dtype_name)).contiguous()
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
                or "loop_adapters" in name
            ) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
def build_sliding_window_causal_mask(
    q_len: int,
    kv_len: int,
    q_offset: int,
    kv_offset: int,
    window_size: int,
    device: torch.device,
) -> Tensor:
    q_idx = torch.arange(q_len, device=device)[:, None] + q_offset
    k_idx = torch.arange(kv_len, device=device)[None, :] + kv_offset
    invalid = (k_idx > q_idx) | (k_idx < (q_idx - window_size + 1))
    mask = torch.zeros((q_len, kv_len), device=device, dtype=torch.float32)
    mask.masked_fill_(invalid, float("-inf"))
    return mask


def expand_kv_heads(x: Tensor, num_heads: int) -> Tensor:
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def shift_expanded_keys(k: Tensor, head_shifts: tuple[int, ...]) -> Tensor:
    shifted = k.clone()
    for head_idx, shift in enumerate(head_shifts):
        if shift <= 0:
            continue
        shifted[:, head_idx, :, :] = 0.0
        if shift < k.size(2):
            shifted[:, head_idx, shift:, :] = k[:, head_idx, :-shift, :]
    return shifted


def pairwise_rotate_last_dim(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class LayerRoPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("LayerRoPE requires an even dimension")
        self.layer_rope_theta = nn.Parameter(torch.zeros(dim // 2, dtype=torch.float32))

    def _cos_sin(self, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        theta = self.layer_rope_theta.to(device=device, dtype=torch.float32)
        return torch.cos(theta).to(dtype=dtype), torch.sin(theta).to(dtype=dtype)

    def rotate_columns(self, weight: Tensor) -> Tensor:
        cos, sin = self._cos_sin(weight.dtype, weight.device)
        return pairwise_rotate_last_dim(weight, cos, sin)


class LowRankAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.out_features = out_features
        self.scale = 1.0 / max(rank, 1)
        if rank > 0:
            self.a = nn.Parameter(torch.empty(rank, in_features))
            self.b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.normal_(self.a, mean=0.0, std=in_features ** -0.5)
        else:
            self.register_parameter("a", None)
            self.register_parameter("b", None)

    def forward(self, x: Tensor, rope: LayerRoPE | None = None) -> Tensor:
        if self.a is None or self.b is None:
            return x.new_zeros((*x.shape[:-1], self.out_features))
        a = self.a.to(device=x.device, dtype=x.dtype)
        if rope is not None:
            a = rope.rotate_columns(a)
        hidden = F.linear(x, a)
        return F.linear(hidden, self.b.to(device=x.device, dtype=x.dtype)) * self.scale

    def effective_weight(self, base_weight: Tensor, dtype: torch.dtype, device: torch.device, rope: LayerRoPE | None = None) -> Tensor:
        weight = base_weight.to(device=device, dtype=dtype)
        if self.a is None or self.b is None:
            return weight
        a = self.a.to(device=device, dtype=dtype)
        if rope is not None:
            a = rope.rotate_columns(a)
        b = self.b.to(device=device, dtype=dtype)
        return weight + (b @ a) * self.scale


class LoopPassAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_hidden: int, rank: int):
        super().__init__()
        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        self.attn_q = LowRankAdapter(dim, dim, rank)
        self.attn_k = LowRankAdapter(dim, kv_dim, rank)
        self.attn_v = LowRankAdapter(dim, kv_dim, rank)
        self.attn_proj = LowRankAdapter(dim, dim, rank)
        self.mlp_up_rope = LayerRoPE(dim)
        self.mlp_gate = LowRankAdapter(dim, mlp_hidden, rank)
        self.mlp_value = LowRankAdapter(dim, mlp_hidden, rank)
        self.mlp_down = LowRankAdapter(mlp_hidden, dim, rank)


class HyperConnection(nn.Module):
    def __init__(self, effective_depth: int, dim: int):
        super().__init__()
        self.effective_depth = effective_depth
        self.hyper_conn_weights = nn.Parameter(torch.ones((effective_depth, effective_depth + 1), dtype=torch.float32))

    def forward(self, layer_idx: int, root: Tensor, history: list[Tensor]) -> Tensor:
        sources = [root, *history]
        weights = self.hyper_conn_weights[layer_idx, : len(sources)].to(device=root.device, dtype=root.dtype)
        stacked = torch.stack(sources, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0)


class SlidingCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int,
        chunk_size: int,
        shift_offsets: tuple[int, ...],
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"SLIDING_WINDOW_SIZE must be positive, got {window_size}")
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.window_size = window_size
        self.chunk_size = chunk_size if chunk_size > 0 else window_size
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.fixed_shift_offsets = shift_offsets

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        vn = F.normalize(v, dim=-1)
        return y - (y * vn).sum(dim=-1, keepdim=True) * vn

    def forward(
        self,
        x: Tensor,
        v_embed: Tensor | None = None,
        adapter: LoopPassAdapter | None = None,
        shift_enabled: bool = False,
        use_xsa_override: bool = False,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        if adapter is not None:
            q_weight = adapter.attn_q.effective_weight(self.c_q.weight, x.dtype, x.device)
            k_weight = adapter.attn_k.effective_weight(self.c_k.weight, x.dtype, x.device)
            v_weight = adapter.attn_v.effective_weight(self.c_v.weight, x.dtype, x.device)
            q = F.linear(x, q_weight)
            k = F.linear(x, k_weight)
            v = F.linear(x, v_weight)
        else:
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = (q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k_full = expand_kv_heads(k, self.num_heads)
        v_full = expand_kv_heads(v, self.num_heads)
        if shift_enabled:
            head_shifts = [0] * self.num_heads
            kv_group_size = self.num_heads // self.num_kv_heads
            shiftable_heads: list[int] = []
            for kv_head_idx in range(self.num_kv_heads):
                group_start = kv_head_idx * kv_group_size
                shiftable_heads.extend(range(group_start + 1, group_start + kv_group_size))
            for head_idx, shift in zip(shiftable_heads, self.fixed_shift_offsets):
                head_shifts[head_idx] = shift
            k_full = shift_expanded_keys(k_full, tuple(head_shifts))
        chunks: list[Tensor] = []
        for qs in range(0, seqlen, self.chunk_size):
            qe = min(qs + self.chunk_size, seqlen)
            ctx_start = max(0, qs - self.window_size + 1)
            q_chunk = q[:, :, qs:qe, :]
            k_chunk = k_full[:, :, ctx_start:qe, :]
            v_chunk = v_full[:, :, ctx_start:qe, :]
            mask = build_sliding_window_causal_mask(
                qe - qs,
                qe - ctx_start,
                qs,
                ctx_start,
                self.window_size,
                x.device,
            )
            chunks.append(
                F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=mask,
                    is_causal=False,
                    enable_gqa=False,
                )
            )
        y = torch.cat(chunks, dim=2)
        if use_xsa_override:
            y = self._xsa_efficient(y, v_full)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        proj_in = y
        if adapter is not None:
            proj_weight = adapter.attn_proj.effective_weight(self.proj.weight, proj_in.dtype, proj_in.device)
            y = F.linear(proj_in, proj_weight)
        else:
            y = self.proj(y)
        return y
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.hidden = hidden
        self.fc_gate = CastedLinear(dim, hidden, bias=False)
        self.fc_value = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, adapter: LoopPassAdapter | None = None) -> Tensor:
        if adapter is not None:
            gate_weight = adapter.mlp_gate.effective_weight(self.fc_gate.weight, x.dtype, x.device)
            value_base = adapter.mlp_up_rope.rotate_columns(self.fc_value.weight.to(device=x.device, dtype=x.dtype))
            value_weight = adapter.mlp_value.effective_weight(value_base, x.dtype, x.device, rope=adapter.mlp_up_rope)
            gate = F.linear(x, gate_weight)
            value = F.linear(x, value_weight)
        else:
            gate = self.fc_gate(x)
            value = self.fc_value(x)
        hidden = F.silu(gate) * value
        if adapter is not None:
            down_weight = adapter.mlp_down.effective_weight(self.proj.weight, hidden.dtype, hidden.device)
            out = F.linear(hidden, down_weight)
        else:
            out = self.proj(hidden)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        window_size: int,
        chunk_size: int,
        shift_offsets: tuple[int, ...],
        dtg: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = SlidingCausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, window_size, chunk_size, shift_offsets
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(
        self,
        x_in: Tensor,
        layer_scale: float,
        v_embed: Tensor | None = None,
        adapter: LoopPassAdapter | None = None,
        shift_enabled: bool = False,
        use_xsa: bool = False,
    ) -> Tensor:
        x_base = x_in
        attn_out = self.attn(
            self.attn_norm(x_base) * layer_scale,
            v_embed=v_embed,
            adapter=adapter,
            shift_enabled=shift_enabled,
            use_xsa_override=use_xsa,
        )
        mlp_out = self.mlp(
            self.mlp_norm(x_base) * layer_scale,
            adapter=adapter,
        )
        x_out = self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = gate * x_out
        return x_out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        sliding_window_size: int = 512,
        sliding_chunk_size: int = 0,
        shifted_attention_layers: int = 3,
        num_prelude_layers: int = 2,
        num_loop_layers: int = 3,
        loop_repeats: int = 2,
        num_epilogue_layers: int = 3,
        lora_rank: int = 8,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        init_seed: int = 1337,
        init_impl: str = "stateless_ortho_v1",
    ):
        super().__init__()
        self.init_seed = init_seed
        self.init_impl = init_impl
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        self.shift_offsets = (1, 2, 3, 4)
        self.shifted_attention_layers = shifted_attention_layers
        self.xsa_last_n = xsa_last_n
        self.ln_scale = ln_scale
        self.num_prelude_layers = num_prelude_layers
        self.num_loop_layers = num_loop_layers
        self.loop_repeats = loop_repeats
        self.num_epilogue_layers = num_epilogue_layers
        self.effective_depth = num_prelude_layers + num_loop_layers * loop_repeats + num_epilogue_layers
        if num_layers != self.effective_depth:
            raise ValueError(
                f"NUM_LAYERS must match looping depth, got {num_layers} vs {self.effective_depth}"
            )
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.hyper_conn = HyperConnection(self.effective_depth, model_dim)
        self.prelude_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    sliding_window_size,
                    sliding_chunk_size,
                    self.shift_offsets,
                    dtg=dtg,
                )
                for _ in range(num_prelude_layers)
            ]
        )
        self.loop_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    sliding_window_size,
                    sliding_chunk_size,
                    self.shift_offsets,
                    dtg=dtg,
                )
                for _ in range(num_loop_layers)
            ]
        )
        mlp_hidden = int(mlp_mult * model_dim)
        self.loop_adapters = nn.ModuleList(
            [
                nn.ModuleList(
                    [LoopPassAdapter(model_dim, num_heads, num_kv_heads, mlp_hidden, lora_rank) for _ in range(loop_repeats)]
                )
                for _ in range(num_loop_layers)
            ]
        )
        self.epilogue_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    sliding_window_size,
                    sliding_chunk_size,
                    self.shift_offsets,
                    dtg=dtg,
                )
                for _ in range(num_epilogue_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self._all_blocks():
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        self._init_weights()

    def _all_blocks(self) -> list[Block]:
        return [*self.prelude_blocks, *self.loop_blocks, *self.epilogue_blocks]

    def _init_weights(self) -> None:
        if self.init_impl not in {"legacy", "stateless_ortho_v1"}:
            raise ValueError(f"Unknown INIT_IMPL={self.init_impl}")
        if self.tie_embeddings:
            if self.init_impl == "legacy":
                nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
            else:
                stateless_normal_(self.tok_emb.weight, self.init_seed, "tok_emb.weight", self.tied_embed_init_std)
        if self.ve_shared is not None:
            if self.init_impl == "legacy":
                nn.init.normal_(self.ve_shared.embed.weight, std=0.01)
            else:
                stateless_normal_(self.ve_shared.embed.weight, self.init_seed, "ve_shared.embed.weight", 0.01)
        num_layers = self.effective_depth
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    if self.init_impl == "legacy":
                        nn.init.orthogonal_(module.weight, gain=1.0)
                    else:
                        stateless_orthogonal_(module.weight, self.init_seed, f"{name}.weight", gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        ve_base = ve_cache["ve"] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        ve_cache: dict = {}
        history: list[Tensor] = []
        layer_idx = 0

        def run_block(block: Block, adapter: LoopPassAdapter | None = None) -> None:
            nonlocal x, layer_idx
            x_in = self.hyper_conn(layer_idx, x0, history)
            layer_scale = 1.0 / math.sqrt(layer_idx + 1) if self.ln_scale else 1.0
            ve = self._get_ve(layer_idx, input_ids, ve_cache)
            shift_enabled = layer_idx < min(self.shifted_attention_layers, self.effective_depth)
            use_xsa = layer_idx >= max(0, self.effective_depth - self.xsa_last_n)
            x = block(
                x_in,
                layer_scale,
                v_embed=ve,
                adapter=adapter,
                shift_enabled=shift_enabled,
                use_xsa=use_xsa,
            )
            history.append(x)
            layer_idx += 1

        for block in self.prelude_blocks:
            run_block(block)
        for repeat_idx in range(self.loop_repeats):
            for block_idx, block in enumerate(self.loop_blocks):
                run_block(block, adapter=self.loop_adapters[block_idx][repeat_idx])
        for block in self.epilogue_blocks:
            run_block(block)
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._forward_hidden(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale
def mixed_quantize_int6(
    state_dict: dict[str, Tensor],
    int6_cats: set[str],
    fp16_name_patterns: tuple[str, ...] = (),
):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if any(pattern in name for pattern in fp16_name_patterns):
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough_fp16"
            continue
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def build_gpt(args: Hyperparameters, mtp_num_heads: int | None = None, mtp_loss_weight: float | None = None) -> GPT:
    return GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads if mtp_num_heads is None else mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight if mtp_loss_weight is None else mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        sliding_window_size=args.sliding_window_size,
        sliding_chunk_size=args.sliding_chunk_size,
        shifted_attention_layers=args.shifted_attention_layers,
        num_prelude_layers=args.num_prelude_layers,
        num_loop_layers=args.num_loop_layers,
        loop_repeats=args.loop_repeats,
        num_epilogue_layers=args.num_epilogue_layers,
        lora_rank=args.lora_rank,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        init_seed=args.init_seed,
        init_impl=args.init_impl,
    )


def clone_state_dict_to_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().to("cpu").contiguous().clone() for name, tensor in state_dict.items()}


def build_init_state_dict(args: Hyperparameters) -> dict[str, Tensor]:
    if args.init_impl == "legacy":
        torch.manual_seed(args.init_seed)
    return clone_state_dict_to_cpu(build_gpt(args).state_dict())


def should_delta_encode(name: str) -> bool:
    return any(
        key in name
        for key in (
            ".attn.c_q.",
            ".attn.c_k.",
            ".attn.c_v.",
            ".mlp.fc_gate.",
            ".mlp.fc_value.",
        )
    )


def make_delta_state_dict(target: dict[str, Tensor], reference: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, str]]:
    out: dict[str, Tensor] = {}
    modes: dict[str, str] = {}
    for name, tensor in target.items():
        ref = reference[name]
        if tensor.is_floating_point() and should_delta_encode(name):
            out[name] = (tensor - ref.to(dtype=tensor.dtype)).contiguous()
            modes[name] = "delta"
        else:
            out[name] = tensor.clone()
            modes[name] = "raw"
    return out, modes


def restore_delta_state_dict(
    payload_state: dict[str, Tensor],
    reference: dict[str, Tensor],
    modes: dict[str, str],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, tensor in payload_state.items():
        if modes.get(name) == "delta":
            out[name] = (reference[name].to(dtype=tensor.dtype) + tensor).contiguous()
        else:
            out[name] = tensor
    return out


def quantize_tensor_by_kind(t: Tensor, kind: str) -> tuple[dict[str, object], int]:
    if kind == "int8":
        q, s = quantize_float_tensor(t)
        return {"kind": "int", "bits": 8, "q": q, "scale": s}, tensor_nbytes(q) + tensor_nbytes(s)
    if kind == "int6":
        q, s = quantize_float_tensor_nbit(t, 6)
        return {"kind": "int", "bits": 6, "q": q, "scale": s}, tensor_nbytes(q) + tensor_nbytes(s)
    if kind == "int4":
        q, s = quantize_float_tensor_nbit(t, 4)
        packed = pack_lowbit_tensor(q, 4)
        return {
            "kind": "int_packed",
            "bits": 4,
            "q": packed,
            "scale": s,
            "shape": list(t.shape),
            "numel": int(t.numel()),
        }, tensor_nbytes(packed) + tensor_nbytes(s)
    if kind == "gptq_int6":
        q, s = quantize_int6_per_row(t)
        return {"kind": "int", "bits": 6, "q": q, "scale": s}, tensor_nbytes(q) + tensor_nbytes(s)
    if kind == "fp8_e4m3":
        codes = quantize_minifloat_tensor(t, exp_bits=4, mant_bits=3)
        return {
            "kind": "minifloat",
            "bits": 8,
            "exp_bits": 4,
            "mant_bits": 3,
            "codes": codes,
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
        }, tensor_nbytes(codes)
    if kind == "fp6_e3m2":
        codes = quantize_minifloat_tensor(t, exp_bits=3, mant_bits=2)
        packed = pack_unsigned_lowbit_tensor(codes, 6)
        return {
            "kind": "minifloat",
            "bits": 6,
            "exp_bits": 3,
            "mant_bits": 2,
            "codes": packed,
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "dtype": str(t.dtype).removeprefix("torch."),
        }, tensor_nbytes(packed)
    raise ValueError(f"Unsupported quantization kind {kind}")


def dequantize_tensor_by_kind(obj: dict[str, object], orig_dtype: torch.dtype) -> Tensor:
    if obj["kind"] == "int":
        q = obj["q"]
        s = obj["scale"]
        if getattr(s, "ndim", 0) > 0:
            return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype).contiguous()
        return (q.float() * float(s.item())).to(orig_dtype).contiguous()
    if obj["kind"] == "int_packed":
        q = unpack_lowbit_tensor(obj["q"], int(obj["bits"]), int(obj["numel"])).view(obj["shape"])
        s = obj["scale"]
        if getattr(s, "ndim", 0) > 0:
            return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype).contiguous()
        return (q.float() * float(s.item())).to(orig_dtype).contiguous()
    if obj["kind"] == "minifloat":
        bits = int(obj["bits"])
        if bits == 6:
            codes = unpack_unsigned_lowbit_tensor(obj["codes"], 6, int(obj["numel"])).view(obj["shape"])
        else:
            codes = obj["codes"]
        return dequantize_minifloat_tensor(codes, obj["shape"], int(obj["exp_bits"]), int(obj["mant_bits"]), obj["dtype"]).to(orig_dtype)
    raise ValueError(f"Unsupported stored tensor kind {obj['kind']}")


SCHEME_DEFS: dict[str, dict[str, object]] = {
    "raw_gptq": {
        "source": "raw",
        "quant": {"attn": "gptq_int6", "mlp": "gptq_int6", "embed": "int8", "other": "int8"},
    },
    "raw_int_mixed": {
        "source": "raw",
        "quant": {"attn": "int6", "mlp": "int6", "embed": "int8", "other": "int8"},
    },
    "delta_attn_int8_mlp_int6": {
        "source": "delta_hybrid",
        "quant": {"attn": "int8", "mlp": "int6", "embed": "int8", "other": "int8"},
    },
    "delta_attn_int6_mlp_int6": {
        "source": "delta_hybrid",
        "quant": {"attn": "int6", "mlp": "int6", "embed": "int8", "other": "int8"},
    },
    "delta_attn_int6_mlp_int4": {
        "source": "delta_hybrid",
        "quant": {"attn": "int6", "mlp": "int4", "embed": "int8", "other": "int8"},
    },
}


def quantize_state_dict_scheme(
    state_dict: dict[str, Tensor],
    scheme_name: str,
    fp16_name_patterns: tuple[str, ...] = (),
) -> tuple[dict[str, object], dict[str, int]]:
    if scheme_name not in SCHEME_DEFS:
        raise ValueError(f"Unknown compression scheme {scheme_name}")
    quant_map = SCHEME_DEFS[scheme_name]["quant"]
    quantized: dict[str, object] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if any(pattern in name for pattern in fp16_name_patterns) or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float()
            stats["payload_bytes"] += tensor_nbytes(passthrough[name])
            continue
        stats["num_float_tensors"] += 1
        cat = _classify_param(name)
        kind = quant_map.get(cat, quant_map["other"])
        quantized[name], payload_bytes = quantize_tensor_by_kind(t, kind)
        stats["payload_bytes"] += payload_bytes
    obj: dict[str, object] = {
        "__quant_format__": "scheme_sweep_v1",
        "scheme_name": scheme_name,
        "quantized": quantized,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_scheme(obj: dict[str, object], template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, payload in obj["quantized"].items():
        out[name] = dequantize_tensor_by_kind(payload, template_sd[name].dtype)
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if flash_attn_3_func is None:
        raise RuntimeError("Neither flash_attn_interface nor flash_attn is available in this environment")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    # Sliding-window attention uses an explicit causal mask, which flash-only SDP
    # cannot handle reliably under torch.compile fake-tensor tracing.
    enable_math_sdp(True)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = build_gpt(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    init_state_cpu = build_init_state_dict(args)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    matrix_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    token_param_ids = {id(base_model.tok_emb.weight)}
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        token_param_ids.add(id(base_model.bigram.embed.weight))
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        token_param_ids.add(id(base_model.ve_shared.embed.weight))
    head_param_ids: set[int] = set()
    if base_model.lm_head is not None:
        head_param_ids.add(id(base_model.lm_head.weight))
    for name, param in base_model.named_parameters():
        if not param.requires_grad or id(param) in token_param_ids or id(param) in head_param_ids:
            continue
        if param.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(param)
        else:
            scalar_params.append(param)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = list(range(max(0, base_model.effective_depth - args.xsa_last_n), base_model.effective_depth))
    shifted_layers = list(range(min(args.shifted_attention_layers, base_model.effective_depth)))
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(
        f"looping_layout prelude:{args.num_prelude_layers} loop_layers:{args.num_loop_layers} "
        f"loop_repeats:{args.loop_repeats} epilogue:{args.num_epilogue_layers} "
        f"effective_depth:{base_model.effective_depth} lora_rank:{args.lora_rank}"
    )
    log0(f"shifted_attention_layers:{shifted_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=True")
    log0(
        f"attention_mode:fixed_sliding_shifted num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"sliding_window_size:{args.sliding_window_size} "
        f"sliding_chunk_size:{args.sliding_chunk_size if args.sliding_chunk_size > 0 else args.sliding_window_size}"
    )
    log0("m07:unified_hyper_connection enabled")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    log0(f"init_seed:{args.init_seed} init_impl:{args.init_impl}")
    log0(f"export_mode:{args.export_mode}")
    log0(f"compression_schemes:{','.join(args.compression_schemes)}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply EMA weights (better than SWA alone per PR#401)
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = clone_state_dict_to_cpu(export_sd)
    delta_payload_cpu, delta_modes = make_delta_state_dict(sd_cpu, init_state_cpu)
    compression_results: list[tuple[str, str]] = []
    for scheme_name in args.compression_schemes:
        try:
            scheme_def = SCHEME_DEFS.get(scheme_name)
            if scheme_def is None:
                raise ValueError(f"Unknown compression scheme {scheme_name}")
            source_kind = str(scheme_def["source"])
            if source_kind == "delta_hybrid":
                payload_state = delta_payload_cpu
                payload_modes = delta_modes
            elif source_kind == "raw":
                payload_state = sd_cpu
                payload_modes = {name: "raw" for name in sd_cpu}
            else:
                raise ValueError(f"Unknown source kind {source_kind}")

            quant_obj, quant_stats = quantize_state_dict_scheme(
                payload_state,
                scheme_name,
                fp16_name_patterns=args.delta_fp16_name_patterns,
            )
            quant_obj["delta_modes"] = payload_modes
            quant_obj["source_kind"] = source_kind
            if source_kind == "delta_hybrid":
                quant_obj["init_seed"] = args.init_seed
                quant_obj["init_impl"] = args.init_impl
            quant_buf = io.BytesIO()
            torch.save(quant_obj, quant_buf)
            quant_raw = quant_buf.getvalue()
            quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)
            artifact_name = f"final_model.{scheme_name}.ptz"
            if master_process:
                with open(artifact_name, "wb") as f:
                    f.write(quant_blob)
                quant_file_bytes = len(quant_blob)
                code_bytes = len(code.encode("utf-8"))
                ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["payload_bytes"], 1)
                log0(
                    f"Serialized model {scheme_name}: {quant_file_bytes} bytes "
                    f"(source:{source_kind} payload:{quant_stats['payload_bytes']} raw_torch:{len(quant_raw)} payload_ratio:{ratio:.2f}x)"
                )
                log0(f"Total submission size {scheme_name}: {quant_file_bytes + code_bytes} bytes")
            if distributed:
                dist.barrier()

            quant_state = torch.load(
                io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob)),
                map_location="cpu",
            )
            deq_payload = dequantize_state_dict_scheme(quant_state, payload_state)
            roundtrip_state = restore_delta_state_dict(
                deq_payload,
                init_state_cpu,
                quant_state.get("delta_modes", payload_modes),
            )
            base_model.load_state_dict(roundtrip_state, strict=True)
            torch.cuda.synchronize()
            t_qeval = time.perf_counter()
            q_val_loss, q_val_bpb = eval_val(
                args,
                compiled_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            torch.cuda.synchronize()
            log0(
                f"final_{scheme_name}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
            )
            log0(f"final_{scheme_name}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
            compression_results.append((scheme_name, "ok"))
        except Exception as e:
            log0(f"compression_scheme_failed name:{scheme_name} error_type:{type(e).__name__} error:{e}")
            log0(traceback.format_exc().rstrip())
            compression_results.append((scheme_name, "failed"))
    log0("compression_summary " + " ".join(f"{name}:{status}" for name, status in compression_results))
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
