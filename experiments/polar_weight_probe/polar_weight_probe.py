#!/usr/bin/env python3
"""Probe blockwise polar weight compression on CPU.

This script is intentionally lightweight:
- It always supports a numpy-only toy MLP path for local smoke tests.
- If torch is installed, it can also load torchvision or Hugging Face models.
- The codec is a simple blockwise polar transform with optional Hadamard mixing.

The goal is not to be leaderboard-ready yet. It is a quick harness to answer:
1. How much smaller does a compressed weight payload get?
2. How much error does decode introduce?
3. Does a single inference pass noticeably degrade?
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def smallest_uint_dtype(bits: int) -> np.dtype:
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    raise ValueError(f"Unsupported bit width: {bits}")


def hadamard_transform(blocks: np.ndarray) -> np.ndarray:
    """Apply a normalized Walsh-Hadamard transform along the last axis."""
    output = np.array(blocks, dtype=np.float32, copy=True)
    width = output.shape[-1]
    h = 1
    while h < width:
        step = h * 2
        reshaped = output.reshape(-1, width)
        for start in range(0, width, step):
            left = reshaped[:, start : start + h].copy()
            right = reshaped[:, start + h : start + step].copy()
            reshaped[:, start : start + h] = left + right
            reshaped[:, start + h : start + step] = left - right
        h = step
    output /= math.sqrt(width)
    return output


def interleave(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty((left.shape[0], left.shape[1] * 2), dtype=np.float32)
    output[:, 0::2] = left
    output[:, 1::2] = right
    return output


def quantize_uniform(values: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    dtype = smallest_uint_dtype(bits)
    if hi <= lo:
        return np.zeros(values.shape, dtype=dtype)
    clipped = np.clip(values, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return np.round(scaled * levels).astype(dtype)


def dequantize_uniform(values: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    if hi <= lo:
        return np.full(np.asarray(values).shape, lo, dtype=np.float32)
    return values.astype(np.float32) * ((hi - lo) / levels) + lo


def pack_sign_bits(values: np.ndarray) -> np.ndarray:
    signs = (np.asarray(values) >= 0).astype(np.uint8).ravel()
    return np.packbits(signs, bitorder="little")


def unpack_sign_bits(packed: np.ndarray, count: int, shape: Tuple[int, ...]) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[:count]
    return (bits.astype(np.float32) * 2.0 - 1.0).reshape(shape)


def flatten_numeric_output(obj: Any) -> np.ndarray:
    if hasattr(obj, "logits"):
        obj = obj.logits
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32).ravel()
    if isinstance(obj, dict):
        parts = [flatten_numeric_output(value) for value in obj.values()]
        return np.concatenate([part for part in parts if part.size], dtype=np.float32) if parts else np.array([], dtype=np.float32)
    if isinstance(obj, (list, tuple)):
        parts = [flatten_numeric_output(value) for value in obj]
        return np.concatenate([part for part in parts if part.size], dtype=np.float32) if parts else np.array([], dtype=np.float32)
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().numpy().astype(np.float32).ravel()
    return np.array([], dtype=np.float32)


@dataclass
class TensorCompressionResult:
    payload: Dict[str, Any]
    original_bytes: int
    compressed_bytes: int
    compressed_zlib_bytes: int


@dataclass
class TurboMLPFCProbeCase:
    layer_name: str
    queries: np.ndarray
    fc_weight_rows: np.ndarray
    fc_bias: np.ndarray
    exact_scores: np.ndarray
    exact_output: np.ndarray
    proj_weight: np.ndarray
    proj_bias: np.ndarray
    proj_right_multiply: bool
    act_fn: Any


class PolarWeightCodec:
    def __init__(
        self,
        block_size: int = 8,
        first_level_bits: int = 12,
        upper_level_bits: int = 8,
        use_hadamard: bool = True,
        residual_kind: str = "none",
        residual_scale_bits: int = 8,
        residual_jl_dim: int = 0,
        residual_jl_seed: int = 0,
        restore_dtype: str = "original",
    ) -> None:
        if not power_of_two(block_size):
            raise ValueError("block_size must be a power of two")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if residual_kind not in {"none", "sign", "jl_sign"}:
            raise ValueError(f"Unsupported residual kind: {residual_kind}")
        if residual_scale_bits < 0:
            raise ValueError("residual_scale_bits must be >= 0")
        if residual_jl_dim < 0:
            raise ValueError("residual_jl_dim must be >= 0")
        if restore_dtype not in {"original", "fp16", "fp32"}:
            raise ValueError(f"Unsupported restore dtype: {restore_dtype}")
        self.block_size = block_size
        self.first_level_bits = first_level_bits
        self.upper_level_bits = upper_level_bits
        self.use_hadamard = use_hadamard
        self.residual_kind = residual_kind
        self.residual_scale_bits = residual_scale_bits
        self.residual_jl_dim = residual_jl_dim or block_size
        self.residual_jl_seed = residual_jl_seed
        self.restore_dtype = restore_dtype
        self._jl_projection_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def encode_tensor(self, array: np.ndarray) -> TensorCompressionResult:
        original = np.asarray(array)
        if original.dtype.kind not in "fc":
            payload = {
                "kind": "raw",
                "shape": list(original.shape),
                "dtype": str(original.dtype),
                "data": original.tobytes(),
            }
            return self._result(payload, original.nbytes)

        flat = original.astype(np.float32, copy=False).ravel()
        padding = (-flat.size) % self.block_size
        if padding:
            flat = np.pad(flat, (0, padding))

        blocks = flat.reshape(-1, self.block_size)
        if self.use_hadamard:
            blocks = hadamard_transform(blocks)
        root, first_level, upper_levels, base_blocks = self._encode_polar_blocks(blocks)

        payload = {
            "kind": "polar",
            "shape": list(original.shape),
            "dtype": str(original.dtype),
            "block_size": self.block_size,
            "padding": padding,
            "use_hadamard": self.use_hadamard,
            "first_level_bits": self.first_level_bits,
            "upper_level_bits": self.upper_level_bits,
            "root": root,
            "first_level": first_level,
            "upper_levels": upper_levels,
            "residual_kind": self.residual_kind,
            "restore_dtype": self.restore_dtype,
        }
        if self.residual_kind == "sign":
            residual = blocks - base_blocks
            block_scales = np.mean(np.abs(residual), axis=1)
            payload["residual_signs"] = pack_sign_bits(residual)
            payload["residual_count"] = int(residual.size)
            payload["residual_scale_bits"] = self.residual_scale_bits
            if self.residual_scale_bits > 0:
                scale_max = float(np.max(block_scales)) if block_scales.size else 0.0
                payload["residual_scale_max"] = np.float16(scale_max)
                payload["residual_scale"] = quantize_uniform(block_scales, 0.0, scale_max, self.residual_scale_bits)
            else:
                payload["residual_scale"] = np.float16(float(np.mean(block_scales)) if block_scales.size else 0.0)
        elif self.residual_kind == "jl_sign":
            residual = blocks - base_blocks
            projection = self._jl_projection(self.residual_jl_dim)
            sketch = residual @ projection.T
            gamma = np.linalg.norm(residual, axis=1)
            payload["residual_jl_dim"] = self.residual_jl_dim
            payload["residual_jl_seed"] = self.residual_jl_seed
            payload["residual_signs"] = pack_sign_bits(sketch)
            payload["residual_count"] = int(sketch.size)
            payload["residual_scale_bits"] = self.residual_scale_bits
            if self.residual_scale_bits > 0:
                gamma_max = float(np.max(gamma)) if gamma.size else 0.0
                payload["residual_scale_max"] = np.float16(gamma_max)
                payload["residual_scale"] = quantize_uniform(gamma, 0.0, gamma_max, self.residual_scale_bits)
            else:
                payload["residual_scale"] = np.float16(float(np.mean(gamma)) if gamma.size else 0.0)
        return self._result(payload, original.nbytes)

    def encode_rowwise_matrix(self, matrix_rows: np.ndarray) -> TensorCompressionResult:
        rows = np.asarray(matrix_rows, dtype=np.float32)
        if rows.ndim != 2:
            raise ValueError(f"encode_rowwise_matrix expects a 2D matrix, got shape {rows.shape}")

        row_padding = (-rows.shape[1]) % self.block_size
        if row_padding:
            rows = np.pad(rows, ((0, 0), (0, row_padding)))
        row_blocks = rows.shape[1] // self.block_size
        blocks = rows.reshape(-1, self.block_size)
        if self.use_hadamard:
            blocks = hadamard_transform(blocks)

        root, first_level, upper_levels, base_blocks = self._encode_polar_blocks(blocks)
        payload = {
            "kind": "rowwise_polar",
            "shape": list(matrix_rows.shape),
            "dtype": str(np.asarray(matrix_rows).dtype),
            "block_size": self.block_size,
            "row_padding": row_padding,
            "row_blocks": row_blocks,
            "use_hadamard": self.use_hadamard,
            "first_level_bits": self.first_level_bits,
            "upper_level_bits": self.upper_level_bits,
            "root": root,
            "first_level": first_level,
            "upper_levels": upper_levels,
            "residual_kind": self.residual_kind,
            "restore_dtype": self.restore_dtype,
        }
        if self.residual_kind == "jl_sign":
            residual = blocks - base_blocks
            projection = self._jl_projection(self.residual_jl_dim)
            sketch = residual @ projection.T
            gamma = np.linalg.norm(residual, axis=1)
            payload["residual_jl_dim"] = self.residual_jl_dim
            payload["residual_jl_seed"] = self.residual_jl_seed
            payload["residual_signs"] = pack_sign_bits(sketch)
            payload["residual_count"] = int(sketch.size)
            payload["residual_scale_bits"] = self.residual_scale_bits
            if self.residual_scale_bits > 0:
                gamma_max = float(np.max(gamma)) if gamma.size else 0.0
                payload["residual_scale_max"] = np.float16(gamma_max)
                payload["residual_scale"] = quantize_uniform(gamma, 0.0, gamma_max, self.residual_scale_bits)
            else:
                payload["residual_scale"] = np.float16(float(np.mean(gamma)) if gamma.size else 0.0)
        elif self.residual_kind != "none":
            raise ValueError("encode_rowwise_matrix currently supports residual_kind none or jl_sign only")
        return self._result(payload, np.asarray(matrix_rows).nbytes)

    def decode_tensor(self, payload: Dict[str, Any]) -> np.ndarray:
        if payload["kind"] == "raw":
            array = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).copy()
            return array.reshape(payload["shape"])
        blocks = self._decode_polar_blocks(
            np.asarray(payload["root"]),
            np.asarray(payload["first_level"]),
            [np.asarray(level) for level in payload["upper_levels"]],
            int(payload["first_level_bits"]),
            int(payload["upper_level_bits"]),
        )
        residual_kind = payload.get("residual_kind", "none")
        if residual_kind == "sign":
            signs = unpack_sign_bits(payload["residual_signs"], int(payload["residual_count"]), blocks.shape)
            residual_scale_bits = int(payload.get("residual_scale_bits", 0))
            if residual_scale_bits > 0:
                scales = dequantize_uniform(
                    np.asarray(payload["residual_scale"]),
                    0.0,
                    float(payload["residual_scale_max"]),
                    residual_scale_bits,
                )[:, None]
            else:
                scales = np.full((blocks.shape[0], 1), float(payload["residual_scale"]), dtype=np.float32)
            blocks = blocks + signs * scales
        elif residual_kind == "jl_sign":
            jl_dim = int(payload.get("residual_jl_dim", self.block_size))
            projection = self._jl_projection(jl_dim, int(payload.get("residual_jl_seed", self.residual_jl_seed)))
            signs = unpack_sign_bits(payload["residual_signs"], int(payload["residual_count"]), (blocks.shape[0], jl_dim))
            residual_scale_bits = int(payload.get("residual_scale_bits", 0))
            if residual_scale_bits > 0:
                gamma = dequantize_uniform(
                    np.asarray(payload["residual_scale"]),
                    0.0,
                    float(payload["residual_scale_max"]),
                    residual_scale_bits,
                )[:, None]
            else:
                gamma = np.full((blocks.shape[0], 1), float(payload["residual_scale"]), dtype=np.float32)
            qjl_factor = math.sqrt(math.pi / 2.0) / max(jl_dim, 1)
            blocks = blocks + qjl_factor * gamma * (signs @ projection)
        elif residual_kind != "none":
            raise ValueError(f"Unsupported residual kind: {residual_kind}")

        if payload["use_hadamard"]:
            blocks = hadamard_transform(blocks)

        flat = blocks.reshape(-1)
        if payload["padding"]:
            flat = flat[: -payload["padding"]]
        return flat.reshape(payload["shape"]).astype(self._restore_array_dtype(np.dtype(payload["dtype"])), copy=False)

    def score_rowwise_queries(self, queries: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
        if payload["kind"] != "rowwise_polar":
            raise ValueError(f"Expected rowwise_polar payload, got {payload['kind']}")

        original_shape = tuple(np.asarray(queries).shape)
        if len(original_shape) < 2:
            raise ValueError(f"Queries must have shape [..., input_dim], got {original_shape}")
        input_dim = int(payload["shape"][1])
        if original_shape[-1] != input_dim:
            raise ValueError(f"Query width mismatch: expected {input_dim}, got {original_shape[-1]}")

        flat_queries = np.asarray(queries, dtype=np.float32).reshape(-1, input_dim)
        row_padding = int(payload["row_padding"])
        row_blocks = int(payload["row_blocks"])
        if row_padding:
            flat_queries = np.pad(flat_queries, ((0, 0), (0, row_padding)))
        query_blocks = flat_queries.reshape(flat_queries.shape[0], row_blocks, self.block_size)
        if payload["use_hadamard"]:
            query_blocks = hadamard_transform(query_blocks.reshape(-1, self.block_size)).reshape(query_blocks.shape)

        main_blocks = self._decode_polar_blocks(
            np.asarray(payload["root"]),
            np.asarray(payload["first_level"]),
            [np.asarray(level) for level in payload["upper_levels"]],
            int(payload["first_level_bits"]),
            int(payload["upper_level_bits"]),
        ).reshape(int(payload["shape"][0]), row_blocks, self.block_size)
        scores = np.einsum("qkb,rkb->qr", query_blocks, main_blocks, optimize=True)

        residual_kind = payload.get("residual_kind", "none")
        if residual_kind == "jl_sign":
            jl_dim = int(payload["residual_jl_dim"])
            projection = self._jl_projection(jl_dim, int(payload.get("residual_jl_seed", self.residual_jl_seed)))
            query_sketch = np.matmul(query_blocks, projection.T)
            signs = unpack_sign_bits(
                payload["residual_signs"],
                int(payload["residual_count"]),
                (int(payload["shape"][0]), row_blocks, jl_dim),
            )
            residual_scale_bits = int(payload.get("residual_scale_bits", 0))
            if residual_scale_bits > 0:
                gamma = dequantize_uniform(
                    np.asarray(payload["residual_scale"]),
                    0.0,
                    float(payload["residual_scale_max"]),
                    residual_scale_bits,
                ).reshape(int(payload["shape"][0]), row_blocks)
            else:
                gamma = np.full((int(payload["shape"][0]), row_blocks), float(payload["residual_scale"]), dtype=np.float32)
            qjl_factor = math.sqrt(math.pi / 2.0) / max(jl_dim, 1)
            scores = scores + qjl_factor * np.einsum("qkd,rkd,rk->qr", query_sketch, signs, gamma, optimize=True)
        elif residual_kind != "none":
            raise ValueError(f"Unsupported residual kind for score_rowwise_queries: {residual_kind}")

        return scores.reshape(*original_shape[:-1], int(payload["shape"][0]))

    def _result(self, payload: Dict[str, Any], original_bytes: int) -> TensorCompressionResult:
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized, level=9)
        return TensorCompressionResult(
            payload=payload,
            original_bytes=original_bytes,
            compressed_bytes=len(serialized),
            compressed_zlib_bytes=len(compressed),
        )

    def _decode_polar_blocks(
        self,
        root: np.ndarray,
        first_level_q: np.ndarray,
        upper_level_q: List[np.ndarray],
        first_level_bits: int,
        upper_level_bits: int,
    ) -> np.ndarray:
        current = np.asarray(root, dtype=np.float32)[:, None]
        upper_levels = [
            dequantize_uniform(np.asarray(level), -(math.pi / 4.0), math.pi / 4.0, upper_level_bits) + (math.pi / 4.0)
            for level in upper_level_q
        ]
        first_level = dequantize_uniform(np.asarray(first_level_q), -math.pi, math.pi, first_level_bits)

        for level in reversed(upper_levels):
            left = current * np.cos(level)
            right = current * np.sin(level)
            current = interleave(left, right)

        left = current * np.cos(first_level)
        right = current * np.sin(first_level)
        return interleave(left, right)

    def _encode_polar_blocks(self, blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        angle_levels: List[np.ndarray] = []
        current = np.asarray(blocks, dtype=np.float32)
        for _ in range(int(math.log2(self.block_size))):
            left = current[:, 0::2]
            right = current[:, 1::2]
            angle_levels.append(np.arctan2(right, left))
            current = np.sqrt(left * left + right * right)

        root = current[:, 0].astype(np.float16)
        first_level = quantize_uniform(angle_levels[0], -math.pi, math.pi, self.first_level_bits)
        upper_levels = [
            quantize_uniform(level - (math.pi / 4.0), -(math.pi / 4.0), math.pi / 4.0, self.upper_level_bits)
            for level in angle_levels[1:]
        ]
        decoded = self._decode_polar_blocks(root, first_level, upper_levels, self.first_level_bits, self.upper_level_bits)
        return root, first_level, upper_levels, decoded

    def _restore_array_dtype(self, original_dtype: np.dtype) -> np.dtype:
        if self.restore_dtype == "original":
            return original_dtype
        if self.restore_dtype == "fp16":
            return np.dtype(np.float16)
        if self.restore_dtype == "fp32":
            return np.dtype(np.float32)
        raise ValueError(f"Unsupported restore dtype: {self.restore_dtype}")

    def _jl_projection(self, sketch_dim: int, seed: int | None = None) -> np.ndarray:
        projection_seed = self.residual_jl_seed if seed is None else seed
        key = (sketch_dim, projection_seed)
        if key not in self._jl_projection_cache:
            rng = np.random.default_rng(projection_seed)
            self._jl_projection_cache[key] = rng.standard_normal((sketch_dim, self.block_size), dtype=np.float32)
        return self._jl_projection_cache[key]


def summarize_reconstruction(reference: Dict[str, np.ndarray], restored: Dict[str, np.ndarray]) -> Dict[str, float]:
    sq_error = 0.0
    sq_ref = 0.0
    max_abs = 0.0
    total_values = 0
    for name, ref in reference.items():
        rec = restored[name]
        if ref.dtype.kind not in "fc":
            continue
        ref32 = ref.astype(np.float32, copy=False)
        rec32 = rec.astype(np.float32, copy=False)
        diff = ref32 - rec32
        sq_error += float(np.square(diff).sum())
        sq_ref += float(np.square(ref32).sum())
        max_abs = max(max_abs, float(np.max(np.abs(diff))) if diff.size else 0.0)
        total_values += diff.size
    mse = sq_error / max(total_values, 1)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "relative_l2": math.sqrt(sq_error / max(sq_ref, 1e-12)),
        "max_abs": max_abs,
    }


class ToyMLP:
    def __init__(self, seed: int, width: int, depth: int, input_dim: int, output_dim: int) -> None:
        rng = np.random.default_rng(seed)
        dims = [input_dim] + [width] * depth + [output_dim]
        self.state: Dict[str, np.ndarray] = {}
        for idx, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            scale = 1.0 / math.sqrt(max(fan_in, 1))
            self.state[f"layers.{idx}.weight"] = (rng.standard_normal((fan_out, fan_in), dtype=np.float32) * scale).astype(np.float32)
            self.state[f"layers.{idx}.bias"] = np.zeros((fan_out,), dtype=np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.state.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.state = {name: value.copy() for name, value in state.items()}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = inputs.astype(np.float32)
        layer_count = len(self.state) // 2
        for idx in range(layer_count):
            weight = self.state[f"layers.{idx}.weight"]
            bias = self.state[f"layers.{idx}.bias"]
            hidden = hidden @ weight.T + bias
            if idx != layer_count - 1:
                hidden = np.tanh(hidden)
        return hidden


def build_toy_case(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Any, Any]:
    model = ToyMLP(
        seed=args.seed,
        width=args.toy_width,
        depth=args.toy_depth,
        input_dim=args.toy_input_dim,
        output_dim=args.toy_output_dim,
    )
    rng = np.random.default_rng(args.seed + 1)
    inputs = rng.standard_normal((args.batch_size, args.toy_input_dim), dtype=np.float32)

    def run_model(state: Dict[str, np.ndarray]) -> np.ndarray:
        model.load_state_dict(state)
        return model.forward(inputs)

    return model.state_dict(), inputs, run_model


def maybe_import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for this model source, but it is not installed") from exc


def build_torchvision_case(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Any, Any]:
    torch = maybe_import_torch()
    try:
        import torchvision.models as tv_models  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchvision is required for --source torchvision") from exc

    if not hasattr(tv_models, args.model_id):
        raise ValueError(f"Unknown torchvision model: {args.model_id}")

    builder = getattr(tv_models, args.model_id)
    kwargs = {}
    if args.pretrained:
        kwargs["weights"] = "DEFAULT"
    model = builder(**kwargs).cpu().eval()
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

    def extract() -> Dict[str, np.ndarray]:
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()}

    def run_model(state: Dict[str, np.ndarray]) -> np.ndarray:
        current = model.state_dict()
        for name, tensor in current.items():
            array = state[name]
            current[name] = torch.from_numpy(array).to(dtype=tensor.dtype)
        model.load_state_dict(current, strict=True)
        with torch.no_grad():
            outputs = model(inputs)
        return outputs.detach().cpu().numpy()

    return extract(), inputs, run_model


def build_hf_case(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Any, Any]:
    torch = maybe_import_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required for --source hf") from exc

    model = AutoModelForCausalLM.from_pretrained(args.model_id).cpu().eval()
    if args.prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise RuntimeError(
                    "Tokenizer has no pad_token/eos_token/unk_token. "
                    "Please choose a model with a usable padding token or pass --batch-size 1."
                )

        if getattr(model.config, "pad_token_id",
                   None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        encoded = tokenizer(
            [args.prompt] * args.batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.seq_len,
        )

        if getattr(model.config, "pad_token_id",
                   None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        encoded = tokenizer(
            [args.prompt] * args.batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.seq_len,
        )

        inputs = {name: value.cpu() for name, value in encoded.items()}
    else:
        vocab_size = int(getattr(model.config, "vocab_size", 50257))
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
        inputs = {"input_ids": input_ids}

    def extract() -> Dict[str, np.ndarray]:
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()}

    def run_model(state: Dict[str, np.ndarray]) -> np.ndarray:
        current = model.state_dict()
        for name, tensor in current.items():
            array = state[name]
            current[name] = torch.from_numpy(array).to(dtype=tensor.dtype)
        model.load_state_dict(current, strict=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return flatten_numeric_output(outputs)

    return extract(), inputs, run_model


def build_case(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Any, Any]:
    if args.source == "toy":
        return build_toy_case(args)
    if args.source == "torchvision":
        return build_torchvision_case(args)
    if args.source == "hf":
        return build_hf_case(args)
    raise ValueError(f"Unsupported source: {args.source}")


def compress_state_dict(
    state: Dict[str, np.ndarray],
    codec: PolarWeightCodec,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    stats = {
        "original_bytes": 0,
        "compressed_bytes": 0,
        "compressed_zlib_bytes": 0,
    }
    for name, array in state.items():
        result = codec.encode_tensor(array)
        payloads[name] = result.payload
        stats["original_bytes"] += result.original_bytes
        stats["compressed_bytes"] += result.compressed_bytes
        stats["compressed_zlib_bytes"] += result.compressed_zlib_bytes
    return payloads, stats


def decode_state_dict(
    payloads: Dict[str, Dict[str, Any]],
    codec: PolarWeightCodec,
) -> Dict[str, np.ndarray]:
    return {name: codec.decode_tensor(payload) for name, payload in payloads.items()}


def compare_outputs(reference: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
    ref = flatten_numeric_output(reference)
    rec = flatten_numeric_output(restored)
    if ref.size != rec.size:
        raise ValueError(f"Output shape mismatch: {ref.size} vs {rec.size}")
    diff = ref - rec
    mse = float(np.mean(np.square(diff))) if diff.size else 0.0
    denom = float(np.linalg.norm(ref)) or 1e-12
    cosine = float(np.dot(ref, rec) / (np.linalg.norm(ref) * np.linalg.norm(rec) + 1e-12)) if ref.size else 1.0
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def gelu_new_numpy(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def apply_proj(hidden: np.ndarray, weight: np.ndarray, bias: np.ndarray, right_multiply: bool) -> np.ndarray:
    hidden32 = np.asarray(hidden, dtype=np.float32)
    weight32 = np.asarray(weight, dtype=np.float32)
    bias32 = np.asarray(bias, dtype=np.float32)
    if right_multiply:
        return hidden32 @ weight32 + bias32
    return hidden32 @ weight32.T + bias32


def summarize_metric_pairs(reference_list: List[np.ndarray], restored_list: List[np.ndarray]) -> Dict[str, float]:
    if not reference_list:
        return {"mse": 0.0, "rmse": 0.0, "relative_l2": 0.0, "cosine": 1.0, "max_abs": 0.0}
    reference = np.concatenate([np.asarray(item, dtype=np.float32).ravel() for item in reference_list])
    restored = np.concatenate([np.asarray(item, dtype=np.float32).ravel() for item in restored_list])
    return compare_outputs(reference, restored)


def build_toy_turbo_mlp_fc_cases(args: argparse.Namespace) -> List[TurboMLPFCProbeCase]:
    rng = np.random.default_rng(args.seed)
    input_dim = args.toy_input_dim
    hidden_dim = args.toy_width
    output_dim = args.toy_output_dim

    queries = rng.standard_normal((args.batch_size, input_dim), dtype=np.float32)
    fc_weight_rows = (rng.standard_normal((hidden_dim, input_dim), dtype=np.float32) / math.sqrt(max(input_dim, 1))).astype(np.float32)
    fc_bias = np.zeros((hidden_dim,), dtype=np.float32)
    proj_weight = (rng.standard_normal((output_dim, hidden_dim), dtype=np.float32) / math.sqrt(max(hidden_dim, 1))).astype(np.float32)
    proj_bias = np.zeros((output_dim,), dtype=np.float32)

    exact_scores = queries @ fc_weight_rows.T + fc_bias
    hidden = gelu_new_numpy(exact_scores)
    exact_output = apply_proj(hidden, proj_weight, proj_bias, right_multiply=False)
    return [
        TurboMLPFCProbeCase(
            layer_name="toy.mlp.fc",
            queries=queries,
            fc_weight_rows=fc_weight_rows,
            fc_bias=fc_bias,
            exact_scores=exact_scores,
            exact_output=exact_output,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            proj_right_multiply=False,
            act_fn=gelu_new_numpy,
        )
    ]


def build_hf_turbo_mlp_fc_cases(args: argparse.Namespace) -> List[TurboMLPFCProbeCase]:
    torch = maybe_import_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required for --probe turbo_mlp_fc with --source hf") from exc

    model = AutoModelForCausalLM.from_pretrained(args.model_id).cpu().eval()
    if args.prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise RuntimeError("Tokenizer has no usable padding token")
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        encoded = tokenizer(
            [args.prompt] * args.batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.seq_len,
        )
        inputs = {name: value.cpu() for name, value in encoded.items()}
    else:
        vocab_size = int(getattr(model.config, "vocab_size", 50257))
        inputs = {"input_ids": torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)}

    modules = dict(model.named_modules())
    fc_names = [name for name in modules if name.endswith(".mlp.c_fc")]
    if not fc_names:
        raise RuntimeError("No Hugging Face GPT-style .mlp.c_fc modules found")
    if args.target_layer_index >= 0:
        if args.target_layer_index >= len(fc_names):
            raise ValueError(f"target_layer_index {args.target_layer_index} out of range for {len(fc_names)} layers")
        fc_names = [fc_names[args.target_layer_index]]

    captured: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in fc_names}
    hooks = []

    def register_fc_hooks(layer_name: str) -> None:
        fc_module = modules[layer_name]
        proj_name = layer_name[:-4] + "c_proj"
        proj_module = modules.get(proj_name)
        if proj_module is None:
            raise RuntimeError(f"Could not find matching projection module for {layer_name}")

        def fc_pre_hook(_module: Any, inputs_tuple: Tuple[Any, ...]) -> None:
            captured[layer_name]["queries"] = inputs_tuple[0].detach().cpu().to(dtype=torch.float32).numpy()

        def fc_hook(_module: Any, _inputs_tuple: Tuple[Any, ...], output: Any) -> None:
            captured[layer_name]["scores"] = output.detach().cpu().to(dtype=torch.float32).numpy()

        def proj_hook(_module: Any, _inputs_tuple: Tuple[Any, ...], output: Any) -> None:
            captured[layer_name]["output"] = output.detach().cpu().to(dtype=torch.float32).numpy()

        hooks.append(fc_module.register_forward_pre_hook(fc_pre_hook))
        hooks.append(fc_module.register_forward_hook(fc_hook))
        hooks.append(proj_module.register_forward_hook(proj_hook))

    for name in fc_names:
        register_fc_hooks(name)

    with torch.no_grad():
        _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    cases: List[TurboMLPFCProbeCase] = []
    for layer_name in fc_names:
        proj_name = layer_name[:-4] + "c_proj"
        mlp_name = layer_name.rsplit(".", 1)[0]
        fc_module = modules[layer_name]
        proj_module = modules[proj_name]
        mlp_module = modules[mlp_name]
        queries = captured[layer_name]["queries"].reshape(-1, captured[layer_name]["queries"].shape[-1]).astype(np.float32)
        exact_scores = captured[layer_name]["scores"].reshape(-1, captured[layer_name]["scores"].shape[-1]).astype(np.float32)
        exact_output = captured[layer_name]["output"].reshape(-1, captured[layer_name]["output"].shape[-1]).astype(np.float32)
        fc_weight_rows = fc_module.weight.detach().cpu().numpy().T.astype(np.float32)
        fc_bias = fc_module.bias.detach().cpu().numpy().astype(np.float32)
        proj_weight = proj_module.weight.detach().cpu().numpy().astype(np.float32)
        proj_bias = proj_module.bias.detach().cpu().numpy().astype(np.float32)

        def act_fn(values: np.ndarray, module: Any = mlp_module, torch_mod: Any = torch) -> np.ndarray:
            with torch_mod.no_grad():
                output = module.act(torch_mod.from_numpy(np.asarray(values, dtype=np.float32)))
            return output.detach().cpu().numpy().astype(np.float32)

        cases.append(
            TurboMLPFCProbeCase(
                layer_name=layer_name,
                queries=queries,
                fc_weight_rows=fc_weight_rows,
                fc_bias=fc_bias,
                exact_scores=exact_scores,
                exact_output=exact_output,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                proj_right_multiply=True,
                act_fn=act_fn,
            )
        )
    return cases


def build_turbo_mlp_fc_cases(args: argparse.Namespace) -> List[TurboMLPFCProbeCase]:
    if args.source == "toy":
        return build_toy_turbo_mlp_fc_cases(args)
    if args.source == "hf":
        return build_hf_turbo_mlp_fc_cases(args)
    raise ValueError(f"--probe turbo_mlp_fc does not support --source {args.source}")


def evaluate_turbo_mlp_fc_cases(
    cases: List[TurboMLPFCProbeCase],
    codec: PolarWeightCodec,
) -> Dict[str, Any]:
    if codec.residual_kind not in {"none", "jl_sign"}:
        raise ValueError("Turbo-faithful MLP probe expects residual_kind none or jl_sign")

    score_refs: List[np.ndarray] = []
    score_hats: List[np.ndarray] = []
    output_refs: List[np.ndarray] = []
    output_hats: List[np.ndarray] = []
    per_layer: List[Dict[str, Any]] = []
    size_stats = {
        "original_bytes": 0,
        "compressed_bytes": 0,
        "compressed_zlib_bytes": 0,
    }

    for case in cases:
        result = codec.encode_rowwise_matrix(case.fc_weight_rows)
        approx_scores = codec.score_rowwise_queries(case.queries, result.payload) + case.fc_bias[None, :]
        hidden = case.act_fn(approx_scores)
        approx_output = apply_proj(hidden, case.proj_weight, case.proj_bias, case.proj_right_multiply)
        score_metrics = compare_outputs(case.exact_scores, approx_scores)
        output_metrics = compare_outputs(case.exact_output, approx_output)

        score_refs.append(case.exact_scores)
        score_hats.append(approx_scores)
        output_refs.append(case.exact_output)
        output_hats.append(approx_output)
        size_stats["original_bytes"] += result.original_bytes
        size_stats["compressed_bytes"] += result.compressed_bytes
        size_stats["compressed_zlib_bytes"] += result.compressed_zlib_bytes
        per_layer.append(
            {
                "layer_name": case.layer_name,
                "query_count": int(case.queries.shape[0]),
                "hidden_dim": int(case.fc_weight_rows.shape[0]),
                "input_dim": int(case.fc_weight_rows.shape[1]),
                "size_bytes": {
                    "original_bytes": result.original_bytes,
                    "compressed_bytes": result.compressed_bytes,
                    "compressed_zlib_bytes": result.compressed_zlib_bytes,
                },
                "score": score_metrics,
                "output": output_metrics,
            }
        )

    return {
        "layer_count": len(cases),
        "layers": per_layer,
        "score": summarize_metric_pairs(score_refs, score_hats),
        "mlp_output": summarize_metric_pairs(output_refs, output_hats),
        "size_bytes": {
            **size_stats,
            "ratio_vs_serialized": size_stats["original_bytes"] / max(size_stats["compressed_bytes"], 1),
            "ratio_vs_zlib": size_stats["original_bytes"] / max(size_stats["compressed_zlib_bytes"], 1),
        },
    }


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["roundtrip", "turbo_mlp_fc"], default="roundtrip")
    parser.add_argument("--source", choices=["toy", "torchvision", "hf"], default="hf")
    parser.add_argument("--model-id", default="distilgpt2", help="Model name for torchvision or Hugging Face")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained torchvision weights")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt", default="", help="Optional text prompt for Hugging Face causal LM inputs")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--first-level-bits", type=int, default=12)
    parser.add_argument("--upper-level-bits", type=int, default=8)
    parser.add_argument("--no-hadamard", action="store_true")
    parser.add_argument("--residual-kind", choices=["none", "sign", "jl_sign"], default="none")
    parser.add_argument(
        "--residual-scale-bits",
        type=int,
        default=8,
        help="Per-block residual magnitude bits for residual codecs. Use 0 for one tensor-global magnitude.",
    )
    parser.add_argument(
        "--residual-jl-dim",
        type=int,
        default=0,
        help="JL sketch dimension for --residual-kind jl_sign. Defaults to block size.",
    )
    parser.add_argument(
        "--residual-jl-seed",
        type=int,
        default=0,
        help="Seed for the deterministic JL projection used by --residual-kind jl_sign.",
    )
    parser.add_argument(
        "--restore-dtype",
        choices=["original", "fp16", "fp32"],
        default="original",
        help="Target dtype for the one-shot restored weight arrays.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-layer-index",
        type=int,
        default=-1,
        help="For --probe turbo_mlp_fc on Hugging Face models, restrict to one MLP fc layer index. Default -1 means all.",
    )
    parser.add_argument("--toy-width", type=int, default=128)
    parser.add_argument("--toy-depth", type=int, default=3)
    parser.add_argument("--toy-input-dim", type=int, default=128)
    parser.add_argument("--toy-output-dim", type=int, default=32)
    parser.add_argument("--json", action="store_true", help="Print only the final JSON summary")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    codec = PolarWeightCodec(
        block_size=args.block_size,
        first_level_bits=args.first_level_bits,
        upper_level_bits=args.upper_level_bits,
        use_hadamard=not args.no_hadamard,
        residual_kind=args.residual_kind,
        residual_scale_bits=args.residual_scale_bits,
        residual_jl_dim=args.residual_jl_dim,
        residual_jl_seed=args.residual_jl_seed,
        restore_dtype=args.restore_dtype,
    )

    if args.probe == "roundtrip":
        start = time.time()
        state, _inputs, run_model = build_case(args)
        source_build_seconds = time.time() - start

        reference_output = run_model(state)
        payloads, size_stats = compress_state_dict(state, codec)
        restored_state = decode_state_dict(payloads, codec)
        restored_output = run_model(restored_state)

        reconstruction = summarize_reconstruction(state, restored_state)
        output_metrics = compare_outputs(reference_output, restored_output)

        summary = {
            "probe": args.probe,
            "source": args.source,
            "model_id": args.model_id if args.source != "toy" else "toy_mlp",
            "build_seconds": round(source_build_seconds, 4),
            "codec": {
                "block_size": args.block_size,
                "first_level_bits": args.first_level_bits,
                "upper_level_bits": args.upper_level_bits,
                "use_hadamard": not args.no_hadamard,
                "residual_kind": args.residual_kind,
                "residual_scale_bits": args.residual_scale_bits if args.residual_kind != "none" else 0,
                "residual_jl_dim": codec.residual_jl_dim if args.residual_kind == "jl_sign" else 0,
                "residual_jl_seed": args.residual_jl_seed if args.residual_kind == "jl_sign" else 0,
                "restore_dtype": args.restore_dtype,
            },
            "size_bytes": {
                **size_stats,
                "ratio_vs_serialized": size_stats["original_bytes"] / max(size_stats["compressed_bytes"], 1),
                "ratio_vs_zlib": size_stats["original_bytes"] / max(size_stats["compressed_zlib_bytes"], 1),
            },
            "reconstruction": reconstruction,
            "output": output_metrics,
            "tensor_count": len(state),
        }
    else:
        start = time.time()
        cases = build_turbo_mlp_fc_cases(args)
        source_build_seconds = time.time() - start
        turbo_eval = evaluate_turbo_mlp_fc_cases(cases, codec)
        summary = {
            "probe": args.probe,
            "source": args.source,
            "model_id": args.model_id if args.source != "toy" else "toy_mlp_fc",
            "build_seconds": round(source_build_seconds, 4),
            "codec": {
                "block_size": args.block_size,
                "first_level_bits": args.first_level_bits,
                "upper_level_bits": args.upper_level_bits,
                "use_hadamard": not args.no_hadamard,
                "residual_kind": args.residual_kind,
                "residual_scale_bits": args.residual_scale_bits if args.residual_kind != "none" else 0,
                "residual_jl_dim": codec.residual_jl_dim if args.residual_kind == "jl_sign" else 0,
                "residual_jl_seed": args.residual_jl_seed if args.residual_kind == "jl_sign" else 0,
                "restore_dtype": args.restore_dtype,
                "target_layer_index": args.target_layer_index,
            },
            **turbo_eval,
        }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("Interpretation:")
        print(f"- Probe: {args.probe}")
        print(f"- Raw tensor bytes: {summary['size_bytes']['original_bytes']:,}")
        print(f"- Codec payload bytes: {summary['size_bytes']['compressed_bytes']:,}")
        print(f"- Codec+zlib bytes: {summary['size_bytes']['compressed_zlib_bytes']:,}")
        print(f"- Restore dtype: {args.restore_dtype}")
        if args.residual_kind != "none":
            residual_line = f"- Residual: {args.residual_kind} (magnitude bits: {args.residual_scale_bits}"
            if args.residual_kind == "jl_sign":
                residual_line += f", jl_dim: {codec.residual_jl_dim}, jl_seed: {args.residual_jl_seed}"
            residual_line += ")"
            print(residual_line)
        if args.probe == "roundtrip":
            print(f"- Weight relative L2 error: {summary['reconstruction']['relative_l2']:.6f}")
            print(f"- Output relative L2 error: {summary['output']['relative_l2']:.6f}")
        else:
            print(f"- MLP fc score relative L2 error: {summary['score']['relative_l2']:.6f}")
            print(f"- MLP output relative L2 error: {summary['mlp_output']['relative_l2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
