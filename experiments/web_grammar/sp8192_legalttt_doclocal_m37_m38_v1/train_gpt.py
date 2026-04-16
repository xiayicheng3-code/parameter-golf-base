from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import glob
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F

import record_base as base


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
IGNORE_INDEX = -100

ORIG_VALIDATION_DATA = base.ValidationData
ORIG_SEQUENCE_LOADER = base.ShuffledSequenceLoader
ORIG_EVAL_VAL = base.eval_val
ORIG_EVAL_VAL_SLIDING = base.eval_val_sliding
ORIG_EVAL_VAL_TTT = base.eval_val_ttt


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


base.Hyperparameters.doc_local_windows = env_bool("DOC_LOCAL_WINDOWS", True)
base.Hyperparameters.shuffle_docs = env_bool("SHUFFLE_DOCS", True)
base.Hyperparameters.train_context_burnin = int(os.environ.get("TRAIN_CONTEXT_BURNIN", "64"))
base.Hyperparameters.train_token_limit = int(os.environ.get("TRAIN_TOKEN_LIMIT", "0"))
base.Hyperparameters.val_token_limit = int(os.environ.get("VAL_TOKEN_LIMIT", "0"))
base.Hyperparameters.train_doc_limit = int(os.environ.get("TRAIN_DOC_LIMIT", "0"))
base.Hyperparameters.val_doc_limit = int(os.environ.get("VAL_DOC_LIMIT", "0"))


@dataclass(frozen=True)
class ShardSpec:
    path: str
    global_start: int
    num_tokens: int


@dataclass(frozen=True)
class DocSpan:
    doc_id: int
    start: int
    stop: int
    has_closing_bos: bool

    @property
    def raw_len(self) -> int:
        return self.stop - self.start

    @property
    def pred_len(self) -> int:
        return self.raw_len - 1


def shard_token_count(path: Path) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    return int(header[2])


def build_shard_specs(pattern: str) -> tuple[list[ShardSpec], int]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    specs: list[ShardSpec] = []
    total = 0
    for path in files:
        count = shard_token_count(path)
        specs.append(ShardSpec(str(path), total, count))
        total += count
    if total < 2:
        raise ValueError(f"Need at least two tokens from {pattern}")
    return specs, total


class MemmapTokenStore:
    def __init__(self, specs: list[ShardSpec]):
        self.specs = specs
        self.starts = [spec.global_start for spec in specs]
        self.ends = [spec.global_start + spec.num_tokens for spec in specs]
        self.total_tokens = self.ends[-1] if self.ends else 0
        self.arrays = [
            np.memmap(spec.path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(spec.num_tokens,))
            for spec in specs
        ]

    def read_span(self, start: int, end: int) -> np.ndarray:
        if not (0 <= start < end <= self.total_tokens):
            raise ValueError(f"Invalid span [{start}, {end}) for total_tokens={self.total_tokens}")
        out = np.empty((end - start,), dtype=np.uint16)
        cursor = 0
        shard_idx = max(bisect_right(self.starts, start) - 1, 0)
        pos = start
        while pos < end:
            shard_start = self.starts[shard_idx]
            shard_end = self.ends[shard_idx]
            take = min(end, shard_end) - pos
            local_start = pos - shard_start
            out[cursor : cursor + take] = self.arrays[shard_idx][local_start : local_start + take]
            pos += take
            cursor += take
            shard_idx += 1
        return out


def scan_bos_positions(specs: list[ShardSpec], bos_id: int) -> np.ndarray:
    starts: list[int] = []
    for spec in specs:
        arr = np.memmap(spec.path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(spec.num_tokens,))
        rel = np.flatnonzero(arr == bos_id)
        if rel.size:
            starts.extend((spec.global_start + rel.astype(np.int64)).tolist())
    starts = sorted(set(int(x) for x in starts if x >= 0))
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return np.asarray(starts, dtype=np.int64)


def build_doc_spans(
    specs: list[ShardSpec],
    bos_id: int,
    doc_limit: int,
    token_limit: int,
    keep_last_open_doc: bool,
) -> list[DocSpan]:
    bos_positions = scan_bos_positions(specs, bos_id)
    total_tokens = specs[-1].global_start + specs[-1].num_tokens
    spans: list[DocSpan] = []
    for doc_id, start_pos in enumerate(bos_positions.tolist()):
        if token_limit > 0 and start_pos >= token_limit:
            break
        if doc_limit > 0 and len(spans) >= doc_limit:
            break
        if doc_id + 1 < len(bos_positions):
            stop = min(int(bos_positions[doc_id + 1]) + 1, total_tokens)
            has_closing_bos = True
        else:
            if not keep_last_open_doc:
                break
            stop = total_tokens
            has_closing_bos = False
        if stop - start_pos >= 2:
            spans.append(DocSpan(doc_id=doc_id, start=int(start_pos), stop=int(stop), has_closing_bos=has_closing_bos))
    return spans


def phase_for_doc(doc_id: int, seed: int, epoch: int) -> int:
    base_seed = ((doc_id * 0x9E3779B1) ^ seed) & 0xFFFFFFFF
    return int((base_seed + epoch) & 1)


def train_window_starts(pred_len: int, seq_len: int, doc_id: int, seed: int, epoch: int) -> list[int]:
    if pred_len <= seq_len:
        return [0]
    phase = phase_for_doc(doc_id, seed, epoch)
    if pred_len <= 2 * seq_len:
        return [0] if phase == 0 else [pred_len - seq_len]
    num_windows = max(pred_len // seq_len, 1)
    phase_offset = 0 if phase == 0 else pred_len - num_windows * seq_len
    return [phase_offset + idx * seq_len for idx in range(num_windows)]


def eval_window_specs(pred_len: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    if pred_len <= 0:
        return []
    if pred_len <= seq_len:
        return [(0, pred_len, 0, pred_len)]
    starts = [0]
    final_start = pred_len - seq_len
    cursor = stride
    while cursor < final_start:
        starts.append(cursor)
        cursor += stride
    if starts[-1] != final_start:
        starts.append(final_start)
    specs: list[tuple[int, int, int, int]] = []
    scored_until = 0
    for start in starts:
        valid_len = min(seq_len, pred_len - start)
        window_end = start + valid_len
        new_start = max(scored_until, start)
        score_count = max(window_end - new_start, 0)
        score_from = max(new_start - start, 0)
        if score_count > 0:
            specs.append((start, valid_len, score_from, score_count))
            scored_until = window_end
    return specs


def pack_train_window(
    doc_tokens: np.ndarray,
    start: int,
    valid_len: int,
    seq_len: int,
    pad_id: int,
    train_context_burnin: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((seq_len,), pad_id, dtype=np.int64)
    y = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
    x[:valid_len] = doc_tokens[start : start + valid_len].astype(np.int64, copy=False)
    y[:valid_len] = doc_tokens[start + 1 : start + 1 + valid_len].astype(np.int64, copy=False)
    if start > 0 and train_context_burnin > 0:
        burn = min(valid_len, train_context_burnin)
        y[:burn] = IGNORE_INDEX
    return x, y


def pack_eval_window(
    doc_tokens: np.ndarray,
    start: int,
    valid_len: int,
    score_from: int,
    score_count: int,
    seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((seq_len,), pad_id, dtype=np.int64)
    y = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
    x[:valid_len] = doc_tokens[start : start + valid_len].astype(np.int64, copy=False)
    y[score_from : score_from + score_count] = doc_tokens[
        start + 1 + score_from : start + 1 + score_from + score_count
    ].astype(np.int64, copy=False)
    return x, y


class ValidationData:
    def __init__(self, h, device: torch.device):
        if not h.doc_local_windows:
            self._flat = ORIG_VALIDATION_DATA(h, device)
            self.__dict__.update(self._flat.__dict__)
            return
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
        self.bos_id = int(self.sp.bos_id())
        if self.bos_id < 0:
            raise ValueError("Tokenizer must define a BOS token for document-local sampling")
        self.pad_id = int(self.sp.pad_id())
        if self.pad_id < 0:
            self.pad_id = 0
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = base.build_sentencepiece_luts(
            self.sp,
            h.vocab_size,
            device,
        )
        self.train_specs, self.train_total_tokens = build_shard_specs(h.train_files)
        self.val_specs, self.val_total_tokens = build_shard_specs(h.val_files)
        all_train_docs = build_doc_spans(
            self.train_specs,
            self.bos_id,
            h.train_doc_limit,
            h.train_token_limit,
            keep_last_open_doc=False,
        )
        all_val_docs = build_doc_spans(
            self.val_specs,
            self.bos_id,
            h.val_doc_limit,
            h.val_token_limit,
            keep_last_open_doc=True,
        )
        if not all_train_docs:
            raise ValueError("No training documents found for document-local sampling")
        if not all_val_docs:
            raise ValueError("No validation documents found for document-local sampling")
        self.all_train_doc_spans = all_train_docs
        self.all_val_doc_spans = all_val_docs
        self.train_doc_spans = all_train_docs[h.rank::h.world_size]
        self.val_doc_spans = all_val_docs[h.rank::h.world_size]
        if not self.train_doc_spans:
            raise ValueError(f"Rank {h.rank} received zero training documents")
        if not self.val_doc_spans:
            raise ValueError(f"Rank {h.rank} received zero validation documents")
        h._bos_id = self.bos_id
        h._pad_id = self.pad_id
        h._train_specs = self.train_specs
        h._val_specs = self.val_specs
        h._train_doc_spans = self.train_doc_spans
        h._val_doc_spans = self.val_doc_spans
        h._all_train_doc_spans = all_train_docs
        h._all_val_doc_spans = all_val_docs


class ShuffledSequenceLoader:
    def __init__(self, h, device: torch.device):
        self.h = h
        self.device = device
        self.fallback = None
        if not h.doc_local_windows:
            self.fallback = ORIG_SEQUENCE_LOADER(h, device)
            return
        self.seq_len = int(h.train_seq_len)
        self.pad_id = int(h._pad_id)
        self.seed = int(h.seed + h.rank)
        self.shuffle_docs = bool(h.shuffle_docs)
        self.train_context_burnin = max(int(h.train_context_burnin), 0)
        self.store = MemmapTokenStore(h._train_specs)
        self.doc_spans = list(h._train_doc_spans)
        self.next_cycle = 0
        self.current_cycle = 0
        self.doc_order = np.empty((0,), dtype=np.int64)
        self.doc_cursor = 0
        self.current_doc_idx = -1
        self.current_window_specs: list[tuple[int, int]] = []
        self.current_window_cursor = 0
        self.cached_doc_idx = -1
        self.cached_doc_tokens: np.ndarray | None = None
        self._start_new_cycle()

    def _start_new_cycle(self) -> None:
        self.current_cycle = self.next_cycle
        self.next_cycle += 1
        self.doc_order = np.arange(len(self.doc_spans), dtype=np.int64)
        if self.shuffle_docs and self.doc_order.size > 1:
            rng = np.random.default_rng(self.seed + 7919 * self.current_cycle)
            rng.shuffle(self.doc_order)
        self.doc_cursor = 0
        self.current_doc_idx = -1
        self.current_window_specs = []
        self.current_window_cursor = 0

    def _doc_tokens(self, doc_idx: int) -> np.ndarray:
        if self.cached_doc_idx != doc_idx or self.cached_doc_tokens is None:
            span = self.doc_spans[doc_idx]
            self.cached_doc_tokens = self.store.read_span(span.start, span.stop)
            self.cached_doc_idx = doc_idx
        return self.cached_doc_tokens

    def _advance_doc(self) -> None:
        while True:
            if self.doc_cursor >= self.doc_order.size:
                self._start_new_cycle()
            doc_idx = int(self.doc_order[self.doc_cursor])
            self.doc_cursor += 1
            span = self.doc_spans[doc_idx]
            window_specs = [
                (start, min(self.seq_len, span.pred_len - start))
                for start in train_window_starts(span.pred_len, self.seq_len, span.doc_id, self.seed, self.current_cycle)
                if span.pred_len - start > 0
            ]
            if window_specs:
                self.current_doc_idx = doc_idx
                self.current_window_specs = window_specs
                self.current_window_cursor = 0
                return

    def _next_window(self) -> tuple[np.ndarray, np.ndarray]:
        if self.current_window_cursor >= len(self.current_window_specs):
            self._advance_doc()
        start, valid_len = self.current_window_specs[self.current_window_cursor]
        self.current_window_cursor += 1
        doc_tokens = self._doc_tokens(self.current_doc_idx)
        return pack_train_window(doc_tokens, start, valid_len, self.seq_len, self.pad_id, self.train_context_burnin)

    def next_batch(self, global_tokens, grad_accum_steps):
        if self.fallback is not None:
            return self.fallback.next_batch(global_tokens, grad_accum_steps)
        device_tokens = global_tokens // (self.h.world_size * grad_accum_steps)
        batch_size = device_tokens // self.seq_len
        if batch_size <= 0:
            raise ValueError("TRAIN_BATCH_TOKENS is too small for document-local loader")
        x = np.full((batch_size, self.seq_len), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, self.seq_len), IGNORE_INDEX, dtype=np.int64)
        for batch_idx in range(batch_size):
            window_x, window_y = self._next_window()
            x[batch_idx] = window_x
            y[batch_idx] = window_y
        return (
            torch.from_numpy(x).to(self.device, non_blocking=True),
            torch.from_numpy(y).to(self.device, non_blocking=True),
        )


class GPT(base.GPT):
    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        if target_ids.dtype != torch.long:
            target_ids = target_ids.long()
        flat_targets = target_ids.reshape(-1)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            flat_targets,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid = (flat_targets != IGNORE_INDEX).to(dtype=per_token.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (per_token * valid).sum() / denom


def iter_eval_batches(h, doc_spans: list[DocSpan], specs: list[ShardSpec], batch_size: int):
    store = MemmapTokenStore(specs)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for span in doc_spans:
        doc_tokens = store.read_span(span.start, span.stop)
        pred_len = int(doc_tokens.size - 1)
        if pred_len <= 0:
            continue
        for start, valid_len, score_from, score_count in eval_window_specs(pred_len, h.eval_seq_len, h.eval_stride):
            window_x, window_y = pack_eval_window(doc_tokens, start, valid_len, score_from, score_count, h.eval_seq_len, h._pad_id)
            xs.append(window_x)
            ys.append(window_y)
            if len(xs) == batch_size:
                yield np.stack(xs), np.stack(ys)
                xs.clear()
                ys.clear()
    if xs:
        yield np.stack(xs), np.stack(ys)


def masked_byte_count(val_data: ValidationData, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    mask = target_ids != IGNORE_INDEX
    prev_ids = input_ids[mask]
    tgt_ids = target_ids[mask]
    token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()


def evaluate_doc_local(h, device: torch.device, val_data: ValidationData, model) -> tuple[float, float]:
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    batch_size = max(local_batch_tokens // h.eval_seq_len, 1)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for x_np, y_np in iter_eval_batches(h, h._val_doc_spans, h._val_specs, batch_size):
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_loss = model(x, y).detach()
            batch_tokens = (y != IGNORE_INDEX).sum().to(dtype=torch.float64)
            loss_sum += batch_loss.to(torch.float64) * batch_tokens
            token_count += batch_tokens
            byte_count += masked_byte_count(val_data, x, y)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return base._loss_bpb(loss_sum, token_count, byte_count)


def eval_val(h, device, val_data, model):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL(h, device, val_data, model)
    return evaluate_doc_local(h, device, val_data, model)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL_SLIDING(h, device, val_data, base_model, batch_seqs=batch_seqs)
    return evaluate_doc_local(h, device, val_data, base_model)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL_TTT(h, device, val_data, base_model, batch_seqs=batch_seqs)
    raise NotImplementedError("Document-local TTT is not ported in this M37/M38-only experiment. Run with TTT_ENABLED=0.")


def submission_source_text() -> str:
    folder = Path(__file__).resolve().parent
    parts = []
    for path in (folder / "train_gpt.py", folder / "record_base.py"):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def train_and_eval(h, device):
    base.random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = base.ValidationData(h, device)
    if h.doc_local_windows:
        base.log(f"train_shards: {len(h._train_specs)} train_docs_local/global: {len(h._train_doc_spans)}/{len(h._all_train_doc_spans)}")
        base.log(f"val_shards: {len(h._val_specs)} val_docs_local/global: {len(h._val_doc_spans)}/{len(h._all_val_doc_spans)}")
        base.log(f"doc_windows: train_seq={h.train_seq_len} eval_seq={h.eval_seq_len} eval_stride={h.eval_stride}")
        base.log(f"train_context_burnin: {h.train_context_burnin}")
    else:
        base.log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
        base.log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    base_model, compiled_model = base.train_model(h, device, val_data)
    torch._dynamo.reset()
    base.timed_eval("pre-quantization post-ema", base.eval_val, h, device, val_data, compiled_model)
    base.serialize(h, base_model, submission_source_text())
    if h.distributed:
        dist.barrier()
    eval_model = base.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_eval_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    base.timed_eval("quantized", base.eval_val, h, device, val_data, compiled_eval_model)
    if h.sliding_window_enabled:
        label = "quantized_doc_rolling" if h.doc_local_windows else "quantized_sliding_window"
        base.timed_eval(label, base.eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        if h.doc_local_windows:
            base.log("ttt:skipped because document-local TTT is not ported in this experiment")
        else:
            del eval_model, compiled_eval_model
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            ttt_model = base.deserialize(h, device)
            if h.num_loops > 0:
                ttt_model.looping_active = True
            base.timed_eval("quantized_ttt", base.eval_val_ttt, h, device, val_data, ttt_model)
            del ttt_model


base.ValidationData = ValidationData
base.ShuffledSequenceLoader = ShuffledSequenceLoader
base.GPT = GPT
base.eval_val = eval_val
base.eval_val_sliding = eval_val_sliding
base.eval_val_ttt = eval_val_ttt
base.train_and_eval = train_and_eval


def main():
    base.main()


if __name__ == "__main__":
    main()
