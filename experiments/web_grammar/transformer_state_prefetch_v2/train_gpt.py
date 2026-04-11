from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import glob
import io
import json
import math
import os
from pathlib import Path
import queue
import random
import threading
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from grammar_features import FEATURE_NAMES, GrammarStateEmbedding, GrammarVocabTables, build_vocab_tables_from_sentencepiece


ROOT = Path(__file__).resolve().parents[3]
HEADER_BYTES = 256 * np.dtype("<i4").itemsize
LOCAL_CHUNK_MASK_CACHE: dict[tuple[str, str, int, int, int], Tensor] = {}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


@dataclass
class Config:
    run_id: str = os.environ.get("RUN_ID", "state_prefetch_v2_smoke")
    data_path: str = os.environ.get("DATA_PATH", str(ROOT / "data/datasets/fineweb10B_sp1024"))
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", str(ROOT / "data/tokenizers/fineweb_1024_bpe.model"))
    vocab_size: int = env_int("VOCAB_SIZE", 1024)
    train_token_limit: int = env_int("TRAIN_TOKEN_LIMIT", 200_000)
    val_token_limit: int = env_int("VAL_TOKEN_LIMIT", 0)
    train_doc_limit: int = env_int("TRAIN_DOC_LIMIT", 0)
    val_doc_limit: int = env_int("VAL_DOC_LIMIT", 0)
    seq_len: int = env_int("SEQ_LEN", 128)
    batch_size: int = env_int("BATCH_SIZE", 8)
    steps: int = env_int("STEPS", 50)
    eval_every: int = env_int("EVAL_EVERY", 25)
    d_model: int = env_int("D_MODEL", 160)
    n_layers: int = env_int("N_LAYERS", 4)
    n_heads: int = env_int("N_HEADS", 4)
    local_attn_layers: int = env_int("LOCAL_ATTN_LAYERS", 2)
    local_window: int = env_int("LOCAL_WINDOW", 64)
    local_attn_chunk: int = env_int("LOCAL_ATTN_CHUNK", 128)
    local_conv_kernel: int = env_int("LOCAL_CONV_KERNEL", 5)
    dropout: float = env_float("DROPOUT", 0.0)
    lr: float = env_float("LR", 3e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 0.01)
    seed: int = env_int("SEED", 1337)
    use_grammar: bool = env_bool("USE_GRAMMAR", True)
    grammar_init_scale: float = env_float("GRAMMAR_INIT_SCALE", 0.02)
    device: str = os.environ.get("DEVICE", "auto")
    num_workers: int = env_int("NUM_WORKERS", max(min(os.cpu_count() or 4, 8) - 1, 0))
    eval_num_workers: int = env_int("EVAL_NUM_WORKERS", 0)
    prefetch_factor: int = env_int("PREFETCH_FACTOR", 4)
    persistent_workers: bool = env_bool("PERSISTENT_WORKERS", True)
    pin_memory: bool = env_bool("PIN_MEMORY", True)
    host_prefetch_batches: int = env_int("HOST_PREFETCH_BATCHES", 2)
    worker_torch_threads: int = env_int("WORKER_TORCH_THREADS", 1)
    train_feature_cache_docs: int = env_int("TRAIN_FEATURE_CACHE_DOCS", 128)
    cache_max_doc_tokens: int = env_int("CACHE_MAX_DOC_TOKENS", 8192)
    eval_stride: int = env_int("EVAL_STRIDE", 64)
    shuffle_docs: bool = env_bool("SHUFFLE_DOCS", True)

    @property
    def train_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_train_*.bin")

    @property
    def val_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_val_*.bin")


@dataclass(frozen=True)
class ShardSpec:
    path: str
    global_start: int
    num_tokens: int


@dataclass(frozen=True)
class SerializedGrammarTables:
    token_class: np.ndarray
    lexical_role: np.ndarray
    shape: np.ndarray
    quote_kind: np.ndarray
    tag_class: np.ndarray
    flags: np.ndarray


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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def serialize_grammar_tables(tables: GrammarVocabTables) -> SerializedGrammarTables:
    return SerializedGrammarTables(
        token_class=tables.token_class.cpu().numpy().astype(np.int16, copy=True),
        lexical_role=tables.lexical_role.cpu().numpy().astype(np.int16, copy=True),
        shape=tables.shape.cpu().numpy().astype(np.int16, copy=True),
        quote_kind=tables.quote_kind.cpu().numpy().astype(np.int16, copy=True),
        tag_class=tables.tag_class.cpu().numpy().astype(np.int16, copy=True),
        flags=tables.flags.cpu().numpy().astype(np.int32, copy=True),
    )


def deserialize_grammar_tables(tables: SerializedGrammarTables) -> GrammarVocabTables:
    return GrammarVocabTables(
        token_class=torch.from_numpy(tables.token_class.astype(np.int64, copy=False)),
        lexical_role=torch.from_numpy(tables.lexical_role.astype(np.int64, copy=False)),
        shape=torch.from_numpy(tables.shape.astype(np.int64, copy=False)),
        quote_kind=torch.from_numpy(tables.quote_kind.astype(np.int64, copy=False)),
        tag_class=torch.from_numpy(tables.tag_class.astype(np.int64, copy=False)),
        flags=torch.from_numpy(tables.flags.astype(np.int64, copy=False)),
    )


def shard_token_count(path: Path) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    return int(header[2])


def build_shard_specs(pattern: str, token_limit: int) -> tuple[list[ShardSpec], int]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    specs: list[ShardSpec] = []
    total = 0
    remaining = token_limit if token_limit > 0 else None
    for path in files:
        count = shard_token_count(path)
        if remaining is not None:
            if remaining <= 0:
                break
            count = min(count, remaining)
            remaining -= count
        if count <= 0:
            break
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
    keep_last_open_doc: bool,
) -> list[DocSpan]:
    bos_positions = scan_bos_positions(specs, bos_id)
    total_tokens = specs[-1].global_start + specs[-1].num_tokens
    spans: list[DocSpan] = []
    max_docs = len(bos_positions) if doc_limit <= 0 else min(len(bos_positions), doc_limit)
    for doc_id in range(max_docs):
        start = int(bos_positions[doc_id])
        if doc_id + 1 < len(bos_positions):
            stop = min(int(bos_positions[doc_id + 1]) + 1, total_tokens)
            has_closing_bos = True
        else:
            if not keep_last_open_doc:
                break
            stop = total_tokens
            has_closing_bos = False
        if stop - start >= 2:
            spans.append(DocSpan(doc_id=doc_id, start=start, stop=stop, has_closing_bos=has_closing_bos))
    return spans


def phase_for_doc(doc_id: int, seed: int, epoch: int) -> int:
    base = ((doc_id * 0x9E3779B1) ^ seed) & 0xFFFFFFFF
    return int((base + epoch) & 1)


def train_window_starts(pred_len: int, seq_len: int, doc_id: int, seed: int, epoch: int) -> list[int]:
    if pred_len <= seq_len:
        return [0]
    phase = phase_for_doc(doc_id, seed, epoch)
    if pred_len <= 2 * seq_len:
        return [0] if phase == 0 else [pred_len - seq_len]
    num_windows = pred_len // seq_len
    if num_windows <= 1:
        return [0]
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


def count_train_windows(doc_spans: list[DocSpan], seq_len: int) -> int:
    total = 0
    for span in doc_spans:
        if span.pred_len <= 0:
            continue
        if span.pred_len <= 2 * seq_len:
            total += 1
        else:
            total += max(span.pred_len // seq_len, 1)
    return total


def count_eval_windows(doc_spans: list[DocSpan], seq_len: int, stride: int) -> int:
    return sum(len(eval_window_specs(span.pred_len, seq_len, stride)) for span in doc_spans if span.pred_len > 0)


class DocumentWindowDataset(IterableDataset):
    def __init__(
        self,
        specs: list[ShardSpec],
        doc_spans: list[DocSpan],
        seq_len: int,
        mode: str,
        epoch: int,
        use_grammar: bool,
        grammar_tables: SerializedGrammarTables | None,
        pad_id: int,
        eval_stride: int,
        shuffle_docs: bool,
        seed: int,
        feature_cache_docs: int,
        cache_max_doc_tokens: int,
    ):
        super().__init__()
        self.specs = specs
        self.doc_spans = doc_spans
        self.seq_len = seq_len
        self.mode = mode
        self.epoch = epoch
        self.use_grammar = use_grammar
        self.grammar_tables = grammar_tables
        self.pad_id = int(pad_id)
        self.eval_stride = int(eval_stride)
        self.shuffle_docs = shuffle_docs
        self.seed = seed
        self.feature_cache_docs = max(int(feature_cache_docs), 0)
        self.cache_max_doc_tokens = max(int(cache_max_doc_tokens), 0)
        self._worker_store: MemmapTokenStore | None = None
        self._worker_builder: GrammarStateEmbedding | None = None
        self._worker_feature_cache: OrderedDict[int, np.ndarray] | None = None

    def _get_worker_state(self) -> tuple[MemmapTokenStore, GrammarStateEmbedding | None, OrderedDict[int, np.ndarray] | None]:
        if self._worker_store is None:
            self._worker_store = MemmapTokenStore(self.specs)
        if self._worker_builder is None and self.use_grammar and self.grammar_tables is not None:
            self._worker_builder = GrammarStateEmbedding(deserialize_grammar_tables(self.grammar_tables), dim=1, init_scale=0.0)
        if self._worker_feature_cache is None and self.feature_cache_docs > 0:
            self._worker_feature_cache = OrderedDict()
        return self._worker_store, self._worker_builder, self._worker_feature_cache

    def _cache_lookup(self, cache: OrderedDict[int, np.ndarray] | None, doc_id: int) -> np.ndarray | None:
        if cache is None:
            return None
        value = cache.get(doc_id)
        if value is None:
            return None
        cache.move_to_end(doc_id)
        return value

    def _cache_store(self, cache: OrderedDict[int, np.ndarray] | None, doc_id: int, pred_len: int, feature_ids: np.ndarray) -> None:
        if cache is None or pred_len > self.cache_max_doc_tokens or self.feature_cache_docs <= 0:
            return
        cache[doc_id] = feature_ids
        cache.move_to_end(doc_id)
        while len(cache) > self.feature_cache_docs:
            cache.popitem(last=False)

    def __iter__(self):
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        store, builder, feature_cache = self._get_worker_state()
        doc_ids = np.arange(len(self.doc_spans), dtype=np.int64)
        local_doc_ids = doc_ids[worker_id::num_workers]
        if local_doc_ids.size == 0:
            return

        cycle = 0
        while True:
            cycle_doc_ids = local_doc_ids.copy()
            if self.shuffle_docs and cycle_doc_ids.size > 1:
                rng = np.random.default_rng(self.seed + 9973 * (self.epoch + cycle) + worker_id)
                rng.shuffle(cycle_doc_ids)

            for doc_idx in cycle_doc_ids.tolist():
                span = self.doc_spans[doc_idx]
                doc_tokens = store.read_span(span.start, span.stop)
                pred_len = int(doc_tokens.size - 1)
                if pred_len <= 0:
                    continue
                feature_ids = self._cache_lookup(feature_cache, span.doc_id)
                if feature_ids is None and builder is not None:
                    doc_input = torch.from_numpy(doc_tokens[:-1].astype(np.int64, copy=False)).view(1, -1)
                    with torch.inference_mode():
                        feature_ids = builder.build_feature_ids(doc_input).squeeze(0).to(dtype=torch.uint8).contiguous().numpy()
                    self._cache_store(feature_cache, span.doc_id, pred_len, feature_ids)
                if self.mode == "train":
                    phase_epoch = self.epoch + cycle
                    window_specs = [
                        (start, min(self.seq_len, pred_len - start), 0, min(self.seq_len, pred_len - start))
                        for start in train_window_starts(pred_len, self.seq_len, span.doc_id, self.seed, phase_epoch)
                    ]
                else:
                    window_specs = eval_window_specs(pred_len, self.seq_len, self.eval_stride)
                feat_dim = 0 if feature_ids is None else int(feature_ids.shape[1])
                for start, valid_len, score_from, score_count in window_specs:
                    if valid_len <= 0 or score_count <= 0:
                        continue
                    x = np.full((self.seq_len,), self.pad_id, dtype=np.uint16)
                    y = np.full((self.seq_len,), self.pad_id, dtype=np.uint16)
                    mask = np.zeros((self.seq_len,), dtype=np.uint8)
                    x[:valid_len] = doc_tokens[start : start + valid_len].astype(np.uint16, copy=False)
                    y[:valid_len] = doc_tokens[start + 1 : start + 1 + valid_len].astype(np.uint16, copy=False)
                    mask[score_from : score_from + score_count] = 1
                    if feature_ids is None:
                        feats = np.empty((self.seq_len, 0), dtype=np.uint8)
                    else:
                        feats = np.zeros((self.seq_len, feat_dim), dtype=np.uint8)
                        feats[:valid_len] = feature_ids[start : start + valid_len]
                    yield x, y, feats, mask
            if self.mode != "train":
                break
            cycle += 1


def collate_windows(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, feats, masks = zip(*batch, strict=True)
    return np.stack(xs), np.stack(ys), np.stack(feats), np.stack(masks)


def local_chunk_mask(query_len: int, history_len: int, window: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (str(device), str(dtype), int(query_len), int(history_len), int(window))
    cached = LOCAL_CHUNK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    kv_len = history_len + query_len
    q_idx = torch.arange(query_len, device=device)[:, None]
    k_idx = torch.arange(kv_len, device=device)[None, :]
    upper = history_len + q_idx
    lower = upper - window + 1
    allowed = (k_idx <= upper) & (k_idx >= lower)
    mask = torch.full((query_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = mask.masked_fill(allowed, 0)
    LOCAL_CHUNK_MASK_CACHE[key] = mask
    return mask


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int,
    chunk_size: int,
    dropout_p: float,
    training: bool,
) -> Tensor:
    _, _, steps, _ = q.shape
    if window <= 0 or window >= steps:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p if training else 0.0,
            is_causal=True,
        )
    chunk = max(int(chunk_size), 1)
    outputs: list[Tensor] = []
    for start in range(0, steps, chunk):
        end = min(start + chunk, steps)
        kv_start = max(0, start - window + 1)
        # Keep exactly the prefix this chunk can legally see.
        kv_end = end
        q_chunk = q[:, :, start:end, :]
        k_chunk = k[:, :, kv_start:kv_end, :]
        v_chunk = v[:, :, kv_start:kv_end, :]
        history_len = start - kv_start
        attn_mask = local_chunk_mask(
            query_len=end - start,
            history_len=history_len,
            window=window,
            device=q.device,
            dtype=q.dtype,
        )
        out_chunk = F.scaled_dot_product_attention(
            q_chunk,
            k_chunk,
            v_chunk,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
        )
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=2)


class HostBatchPrefetcher:
    def __init__(self, loader: DataLoader, pin_memory: bool, max_prefetch: int):
        self.loader = iter(loader)
        self.pin_memory = pin_memory
        self.queue: queue.Queue[object] = queue.Queue(maxsize=max(int(max_prefetch), 1))
        self._sentinel = object()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="host-batch-prefetch", daemon=True)
        self._thread.start()

    def _to_host_tensor(self, value: np.ndarray | Tensor) -> Tensor:
        tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        if self.pin_memory and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        return tensor

    def _run(self) -> None:
        try:
            for batch in self.loader:
                self.queue.put(tuple(self._to_host_tensor(item) for item in batch))
        except BaseException as exc:  # pragma: no cover - surfaced on next()
            self._error = exc
        finally:
            self.queue.put(self._sentinel)

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        item = self.queue.get()
        if item is self._sentinel:
            if self._error is not None:
                raise self._error
            return None
        return item  # type: ignore[return-value]


class DevicePrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device, pin_memory: bool, host_prefetch_batches: int):
        self.use_host_thread = bool(device.type == "cuda" and host_prefetch_batches > 0)
        self.pin_memory = bool(pin_memory and device.type == "cuda")
        self.host = (
            HostBatchPrefetcher(loader, pin_memory=self.pin_memory, max_prefetch=host_prefetch_batches)
            if self.use_host_thread
            else None
        )
        self.loader = None if self.use_host_thread else iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.next_batch: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        self.preload()

    def preload(self) -> None:
        if self.host is not None:
            batch = self.host.next()
            if batch is None:
                self.next_batch = None
                return
        else:
            try:
                raw_batch = next(self.loader)
            except StopIteration:
                self.next_batch = None
                return
            batch = tuple(torch.from_numpy(item) if isinstance(item, np.ndarray) else item for item in raw_batch)
            if self.pin_memory:
                batch = tuple(t.pin_memory() if t.device.type == "cpu" else t for t in batch)
        if self.stream is None:
            self.next_batch = tuple(t.to(self.device) for t in batch)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = tuple(t.to(self.device, non_blocking=True) for t in batch)

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(int(vocab_size), int(sp.vocab_size()))
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


class CausalBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, local_window: int, local_chunk: int, conv_kernel: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("D_MODEL must be divisible by N_HEADS")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.local_window = int(local_window)
        self.local_chunk = int(local_chunk)
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.conv_ln = nn.LayerNorm(dim) if conv_kernel > 1 else None
        self.conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, groups=dim, bias=False) if conv_kernel > 1 else None
        self.conv_proj = nn.Linear(dim, dim, bias=False) if conv_kernel > 1 else None
        self.conv_kernel = int(conv_kernel)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, dim = x.shape
        if self.conv is not None and self.conv_ln is not None and self.conv_proj is not None:
            conv_in = self.conv_ln(x).transpose(1, 2)
            conv_in = F.pad(conv_in, (self.conv_kernel - 1, 0))
            conv_out = self.conv(conv_in).transpose(1, 2)
            x = x + self.dropout(self.conv_proj(conv_out))
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        if 0 < self.local_window < steps:
            y = sliding_window_attention(
                q,
                k,
                v,
                window=self.local_window,
                chunk_size=self.local_chunk,
                dropout_p=self.dropout.p,
                training=self.training,
            )
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, steps, dim)
        x = x + self.dropout(self.proj(y))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class PrefetchStateTransformer(nn.Module):
    def __init__(self, cfg: Config, grammar_tables) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.grammar_adapter = (
            GrammarStateEmbedding(grammar_tables, cfg.d_model, init_scale=cfg.grammar_init_scale)
            if cfg.use_grammar
            else None
        )
        blocks = []
        for layer_idx in range(cfg.n_layers):
            is_local = layer_idx < cfg.local_attn_layers
            blocks.append(
                CausalBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.dropout,
                    local_window=cfg.local_window if is_local else 0,
                    local_chunk=cfg.local_attn_chunk if is_local else 0,
                    conv_kernel=cfg.local_conv_kernel if is_local else 0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, grammar_feature_ids: Tensor | None = None) -> Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        _, steps = input_ids.shape
        positions = torch.arange(steps, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        if self.grammar_adapter is not None:
            if grammar_feature_ids is not None and grammar_feature_ids.size(-1) > 0:
                x = x + self.grammar_adapter.embed_feature_ids(grammar_feature_ids).to(dtype=x.dtype)
            else:
                x = x + self.grammar_adapter(input_ids).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return F.linear(x, self.tok_emb.weight)


def configure_worker_process(_: int) -> None:
    threads = max(int(os.environ.get("WORKER_TORCH_THREADS", "1")), 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def make_loader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    drop_last: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_windows,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
        kwargs["worker_init_fn"] = configure_worker_process
    return DataLoader(**kwargs)


def masked_cross_entropy(logits: Tensor, target_ids: Tensor, loss_mask: Tensor) -> tuple[Tensor, Tensor]:
    if target_ids.dtype != torch.long:
        target_ids = target_ids.long()
    per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none").view_as(target_ids)
    mask = loss_mask.to(dtype=per_token.dtype)
    token_count = mask.sum().clamp(min=1.0)
    loss_sum = (per_token * mask).sum()
    return loss_sum, token_count


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    base_bytes: Tensor,
    has_leading_space: Tensor,
    is_boundary: Tensor,
    host_prefetch_batches: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    prefetcher = DevicePrefetcher(
        loader,
        device,
        pin_memory=device.type == "cuda",
        host_prefetch_batches=host_prefetch_batches,
    )
    while True:
        batch = prefetcher.next()
        if batch is None:
            break
        x, y, feats, loss_mask = batch
        logits = model(x, grammar_feature_ids=feats if feats.size(-1) > 0 else None)
        loss_sum, token_count = masked_cross_entropy(logits, y, loss_mask)
        total_loss += float(loss_sum.item())
        total_tokens += int(token_count.item())
        prev_ids = x.reshape(-1).long()
        tgt_ids = y.reshape(-1).long()
        flat_mask = loss_mask.reshape(-1).bool()
        token_bytes = base_bytes[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space[tgt_ids] & ~is_boundary[prev_ids]).to(dtype=torch.int16)
        total_bytes += float(token_bytes[flat_mask].to(torch.float64).sum().item())
    model.train()
    val_loss = total_loss / max(float(total_tokens), 1.0)
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = total_tokens / max(total_bytes, 1.0)
    return val_loss, bits_per_token * tokens_per_byte


def export_compressed_model(model: nn.Module, folder: Path, run_id: str) -> tuple[int, int, int, Path]:
    state = {
        key: (value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu())
        for key, value in model.state_dict().items()
    }
    buf = io.BytesIO()
    torch.save(state, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    safe_run_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_id)
    out_path = folder / f"{safe_run_id}.fp16.ptz"
    out_path.write_bytes(blob)
    code_bytes = sum(path.stat().st_size for path in folder.glob("*.py"))
    return len(blob), code_bytes, len(blob) + code_bytes, out_path


def main() -> None:
    cfg = Config()
    folder = Path(__file__).resolve().parent
    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{cfg.run_id}.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as handle:
            print(msg, file=handle)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    if int(sp.vocab_size()) > cfg.vocab_size:
        raise ValueError(f"Tokenizer vocab {sp.vocab_size()} exceeds VOCAB_SIZE={cfg.vocab_size}")
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer must define a BOS token for document boundary recovery")
    pad_id = int(sp.pad_id())
    if pad_id < 0:
        pad_id = 0

    grammar_tables = build_vocab_tables_from_sentencepiece(sp, cfg.vocab_size)
    serialized_grammar_tables = serialize_grammar_tables(grammar_tables) if cfg.use_grammar else None
    train_specs, train_total_tokens = build_shard_specs(cfg.train_files, cfg.train_token_limit)
    val_specs, val_total_tokens = build_shard_specs(cfg.val_files, cfg.val_token_limit)
    train_doc_spans = build_doc_spans(train_specs, bos_id, cfg.train_doc_limit, keep_last_open_doc=False)
    val_doc_spans = build_doc_spans(val_specs, bos_id, cfg.val_doc_limit, keep_last_open_doc=True)
    train_windows = count_train_windows(train_doc_spans, cfg.seq_len)
    val_windows = count_eval_windows(val_doc_spans, cfg.seq_len, cfg.eval_stride)

    def build_train_loader(epoch: int) -> DataLoader:
        dataset = DocumentWindowDataset(
            train_specs,
            train_doc_spans,
            cfg.seq_len,
            mode="train",
            epoch=epoch,
            use_grammar=cfg.use_grammar,
            grammar_tables=serialized_grammar_tables,
            pad_id=pad_id,
            eval_stride=cfg.eval_stride,
            shuffle_docs=cfg.shuffle_docs,
            seed=cfg.seed,
            feature_cache_docs=cfg.train_feature_cache_docs,
            cache_max_doc_tokens=cfg.cache_max_doc_tokens,
        )
        return make_loader(
            dataset,
            cfg.batch_size,
            cfg.num_workers,
            False,
            cfg.prefetch_factor,
            cfg.persistent_workers,
            True,
        )

    val_dataset = DocumentWindowDataset(
        val_specs,
        val_doc_spans,
        cfg.seq_len,
        mode="eval",
        epoch=0,
        use_grammar=cfg.use_grammar,
        grammar_tables=serialized_grammar_tables,
        pad_id=pad_id,
        eval_stride=cfg.eval_stride,
        shuffle_docs=False,
        seed=cfg.seed,
        feature_cache_docs=0,
        cache_max_doc_tokens=0,
    )
    val_loader = make_loader(
        val_dataset,
        cfg.batch_size,
        cfg.eval_num_workers,
        False,
        cfg.prefetch_factor,
        cfg.persistent_workers,
        False,
    )

    base_bytes, has_leading_space, is_boundary = build_sentencepiece_luts(sp, cfg.vocab_size, device)
    model = PrefetchStateTransformer(cfg, grammar_tables).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path.write_text("", encoding="utf-8")
    log(
        json.dumps(
            {
                **cfg.__dict__,
                "device_resolved": str(device),
                "params": sum(p.numel() for p in model.parameters()),
                "bos_id": bos_id,
                "pad_id": pad_id,
                "num_state_features": len(FEATURE_NAMES),
                "doc_only_windows": True,
            },
            indent=2,
        )
    )
    log(
        f"train_tokens:{train_total_tokens} train_docs:{len(train_doc_spans)} train_windows_epoch:{train_windows} "
        f"val_tokens:{val_total_tokens} val_docs:{len(val_doc_spans)} val_windows:{val_windows}"
    )
    log(
        f"loader:num_workers={cfg.num_workers} eval_num_workers={cfg.eval_num_workers} "
        f"host_pin_memory={cfg.pin_memory and device.type == 'cuda'} prefetch_factor={cfg.prefetch_factor} "
        f"persistent_workers={cfg.persistent_workers}"
    )

    train_loader = build_train_loader(epoch=0)
    train_prefetcher = DevicePrefetcher(
        train_loader,
        device,
        cfg.pin_memory,
        host_prefetch_batches=cfg.host_prefetch_batches,
    )
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        batch = train_prefetcher.next()
        if batch is None:
            raise RuntimeError("Training dataset unexpectedly exhausted")
        x, y, feats, loss_mask = batch
        logits = model(x, grammar_feature_ids=feats if feats.size(-1) > 0 else None)
        loss_sum, token_count = masked_cross_entropy(logits, y, loss_mask)
        loss = loss_sum / token_count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step <= 5 or step == cfg.steps or (cfg.eval_every > 0 and step % cfg.eval_every == 0):
            gate = float(model.grammar_adapter.gate.detach().cpu()) if model.grammar_adapter is not None else 0.0
            approx_epoch = ((step - 1) * cfg.batch_size) / max(train_windows, 1)
            log(f"step:{step}/{cfg.steps} approx_epoch:{approx_epoch:.3f} train_loss:{loss.item():.4f} grammar_gate:{gate:.6f}")
        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step != cfg.steps:
            val_loss, val_bpb = evaluate_loader(
                model,
                val_loader,
                device,
                base_bytes,
                has_leading_space,
                is_boundary,
                cfg.host_prefetch_batches,
            )
            log(f"step:{step}/{cfg.steps} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")

    val_loss, val_bpb = evaluate_loader(
        model,
        val_loader,
        device,
        base_bytes,
        has_leading_space,
        is_boundary,
        cfg.host_prefetch_batches,
    )
    artifact_bytes, code_bytes, total_bytes, artifact_path = export_compressed_model(model, folder, cfg.run_id)
    elapsed = time.perf_counter() - t0
    log(f"final val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")
    log(f"artifact_bytes:{artifact_bytes} code_bytes:{code_bytes} total_submission_bytes:{total_bytes}")
    log(f"artifact_path:{artifact_path}")
    log(f"elapsed_seconds:{elapsed:.3f} log_path:{log_path}")


if __name__ == "__main__":
    main()
