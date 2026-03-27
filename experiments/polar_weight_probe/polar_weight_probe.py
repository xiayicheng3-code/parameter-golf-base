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
    clipped = np.clip(values, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    dtype = smallest_uint_dtype(bits)
    return np.round(scaled * levels).astype(dtype)


def dequantize_uniform(values: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return values.astype(np.float32) * ((hi - lo) / levels) + lo


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


class PolarWeightCodec:
    def __init__(
        self,
        block_size: int = 8,
        first_level_bits: int = 12,
        upper_level_bits: int = 8,
        use_hadamard: bool = True,
    ) -> None:
        if not power_of_two(block_size):
            raise ValueError("block_size must be a power of two")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.block_size = block_size
        self.first_level_bits = first_level_bits
        self.upper_level_bits = upper_level_bits
        self.use_hadamard = use_hadamard

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

        angle_levels: List[np.ndarray] = []
        current = blocks
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
        }
        return self._result(payload, original.nbytes)

    def decode_tensor(self, payload: Dict[str, Any]) -> np.ndarray:
        if payload["kind"] == "raw":
            array = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).copy()
            return array.reshape(payload["shape"])

        root = np.asarray(payload["root"], dtype=np.float32)[:, None]
        current = root

        upper_levels = [
            dequantize_uniform(np.asarray(level), -(math.pi / 4.0), math.pi / 4.0, payload["upper_level_bits"]) + (math.pi / 4.0)
            for level in payload["upper_levels"]
        ]
        first_level = dequantize_uniform(
            np.asarray(payload["first_level"]),
            -math.pi,
            math.pi,
            payload["first_level_bits"],
        )

        for level in reversed(upper_levels):
            left = current * np.cos(level)
            right = current * np.sin(level)
            current = interleave(left, right)

        left = current * np.cos(first_level)
        right = current * np.sin(first_level)
        blocks = interleave(left, right)

        if payload["use_hadamard"]:
            blocks = hadamard_transform(blocks)

        flat = blocks.reshape(-1)
        if payload["padding"]:
            flat = flat[: -payload["padding"]]
        return flat.reshape(payload["shape"]).astype(np.dtype(payload["dtype"]), copy=False)

    def _result(self, payload: Dict[str, Any], original_bytes: int) -> TensorCompressionResult:
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized, level=9)
        return TensorCompressionResult(
            payload=payload,
            original_bytes=original_bytes,
            compressed_bytes=len(serialized),
            compressed_zlib_bytes=len(compressed),
        )


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


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=0)
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
    )

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
        "source": args.source,
        "model_id": args.model_id if args.source != "toy" else "toy_mlp",
        "build_seconds": round(source_build_seconds, 4),
        "codec": {
            "block_size": args.block_size,
            "first_level_bits": args.first_level_bits,
            "upper_level_bits": args.upper_level_bits,
            "use_hadamard": not args.no_hadamard,
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

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("Interpretation:")
        print(f"- Raw tensor bytes: {size_stats['original_bytes']:,}")
        print(f"- Codec payload bytes: {size_stats['compressed_bytes']:,}")
        print(f"- Codec+zlib bytes: {size_stats['compressed_zlib_bytes']:,}")
        print(f"- Weight relative L2 error: {reconstruction['relative_l2']:.6f}")
        print(f"- Output relative L2 error: {output_metrics['relative_l2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
