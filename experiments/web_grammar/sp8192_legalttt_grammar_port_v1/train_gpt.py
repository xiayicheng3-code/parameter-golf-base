from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import glob
import io
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import record_base as base
from grammar_features import FEATURE_NAMES, GrammarStateEmbedding, GrammarVocabTables, build_vocab_tables_from_sentencepiece


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
LOCAL_MASK_CACHE: dict[tuple[str, str, int, int, int], Tensor] = {}

ORIG_VALIDATION_DATA = base.ValidationData
ORIG_SEQUENCE_LOADER = base.ShuffledSequenceLoader
ORIG_EVAL_VAL = base.eval_val
ORIG_EVAL_VAL_SLIDING = base.eval_val_sliding
ORIG_EVAL_VAL_TTT = base.eval_val_ttt
ORIG_GPTQ_MIXED_QUANTIZE = base.gptq_mixed_quantize


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


base.Hyperparameters.use_grammar = env_bool("USE_GRAMMAR", True)
base.Hyperparameters.grammar_init_scale = float(os.environ.get("GRAMMAR_INIT_SCALE", "0.0"))
base.Hyperparameters.doc_local_windows = env_bool("DOC_LOCAL_WINDOWS", True)
base.Hyperparameters.train_token_limit = int(os.environ.get("TRAIN_TOKEN_LIMIT", "0"))
base.Hyperparameters.val_token_limit = int(os.environ.get("VAL_TOKEN_LIMIT", "0"))
base.Hyperparameters.train_doc_limit = int(os.environ.get("TRAIN_DOC_LIMIT", "0"))
base.Hyperparameters.val_doc_limit = int(os.environ.get("VAL_DOC_LIMIT", "0"))
base.Hyperparameters.num_workers = int(os.environ.get("NUM_WORKERS", str(max(min(os.cpu_count() or 4, 8) - 1, 0))))
base.Hyperparameters.eval_num_workers = int(os.environ.get("EVAL_NUM_WORKERS", "0"))
base.Hyperparameters.prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "4"))
base.Hyperparameters.persistent_workers = env_bool("PERSISTENT_WORKERS", True)
base.Hyperparameters.pin_memory = env_bool("PIN_MEMORY", True)
base.Hyperparameters.host_prefetch_batches = int(os.environ.get("HOST_PREFETCH_BATCHES", "2"))
base.Hyperparameters.worker_torch_threads = int(os.environ.get("WORKER_TORCH_THREADS", "1"))
base.Hyperparameters.train_feature_cache_docs = int(os.environ.get("TRAIN_FEATURE_CACHE_DOCS", "128"))
base.Hyperparameters.cache_max_doc_tokens = int(os.environ.get("CACHE_MAX_DOC_TOKENS", "8192"))
base.Hyperparameters.local_attn_layers = int(os.environ.get("LOCAL_ATTN_LAYERS", "0"))
base.Hyperparameters.local_window = int(os.environ.get("LOCAL_WINDOW", "128"))
base.Hyperparameters.local_attn_chunk = int(os.environ.get("LOCAL_ATTN_CHUNK", "128"))
base.Hyperparameters.local_conv_kernel = int(os.environ.get("LOCAL_CONV_KERNEL", "0"))
base.Hyperparameters.shuffle_docs = env_bool("SHUFFLE_DOCS", True)


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


def serialize_grammar_tables(tables: GrammarVocabTables) -> SerializedGrammarTables:
    return SerializedGrammarTables(
        token_class=tables.token_class.cpu().numpy().astype(np.int16, copy=True),
        lexical_role=tables.lexical_role.cpu().numpy().astype(np.int16, copy=True),
        shape=tables.shape.cpu().numpy().astype(np.int16, copy=True),
        quote_kind=tables.quote_kind.cpu().numpy().astype(np.int16, copy=True),
        tag_class=tables.tag_class.cpu().numpy().astype(np.int16, copy=True),
        flags=tables.flags.cpu().numpy().astype(np.int64, copy=True),
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
        self.seq_len = int(seq_len)
        self.mode = mode
        self.epoch = int(epoch)
        self.use_grammar = use_grammar
        self.grammar_tables = grammar_tables
        self.pad_id = int(pad_id)
        self.eval_stride = int(eval_stride)
        self.shuffle_docs = shuffle_docs
        self.seed = int(seed)
        self.feature_cache_docs = max(int(feature_cache_docs), 0)
        self.cache_max_doc_tokens = max(int(cache_max_doc_tokens), 0)
        self._worker_store: MemmapTokenStore | None = None
        self._worker_builder: GrammarStateEmbedding | None = None
        self._worker_feature_cache: OrderedDict[int, np.ndarray] | None = None

    def _get_worker_state(self) -> tuple[MemmapTokenStore, GrammarStateEmbedding | None, OrderedDict[int, np.ndarray] | None]:
        if self._worker_store is None:
            self._worker_store = MemmapTokenStore(self.specs)
        if self._worker_builder is None and self.use_grammar and self.grammar_tables is not None:
            tables = deserialize_grammar_tables(self.grammar_tables)
            self._worker_builder = GrammarStateEmbedding(tables, dim=1, init_scale=0.0)
        if self._worker_feature_cache is None and self.feature_cache_docs > 0:
            self._worker_feature_cache = OrderedDict()
        return self._worker_store, self._worker_builder, self._worker_feature_cache

    def _cache_lookup(self, cache: OrderedDict[int, np.ndarray] | None, doc_id: int) -> np.ndarray | None:
        if cache is None:
            return None
        value = cache.get(doc_id)
        if value is not None:
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
                rng = np.random.default_rng(self.seed + 7919 * (self.epoch + cycle) + worker_id)
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


def collate_windows(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, feats, masks = zip(*batch, strict=True)
    return np.stack(xs), np.stack(ys), np.stack(feats), np.stack(masks)


def local_chunk_mask(query_len: int, history_len: int, window: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (str(device), str(dtype), int(query_len), int(history_len), int(window))
    cached = LOCAL_MASK_CACHE.get(key)
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
    LOCAL_MASK_CACHE[key] = mask
    return mask


def sliding_window_attention(q: Tensor, k: Tensor, v: Tensor, window: int, chunk_size: int, training: bool) -> Tensor:
    _, _, steps, _ = q.shape
    if window <= 0 or window >= steps:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    outputs: list[Tensor] = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, steps, chunk):
        end = min(start + chunk, steps)
        kv_start = max(0, start - window + 1)
        kv_end = end
        q_chunk = q[:, :, start:end, :]
        k_chunk = k[:, :, kv_start:kv_end, :]
        v_chunk = v[:, :, kv_start:kv_end, :]
        attn_mask = local_chunk_mask(end - start, start - kv_start, window, q.device, q.dtype)
        outputs.append(
            F.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        )
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
        except BaseException as exc:
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
        self.pin_memory = bool(pin_memory and device.type == "cuda")
        self.use_host_thread = bool(device.type == "cuda" and host_prefetch_batches > 0)
        self.host = HostBatchPrefetcher(loader, pin_memory=self.pin_memory, max_prefetch=host_prefetch_batches) if self.use_host_thread else None
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


def configure_worker_process(_: int) -> None:
    threads = max(int(os.environ.get("WORKER_TORCH_THREADS", "1")), 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def make_loader(dataset: IterableDataset, batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool, drop_last: bool) -> DataLoader:
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
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = base.build_sentencepiece_luts(self.sp, h.vocab_size, device)
        self.grammar_tables = build_vocab_tables_from_sentencepiece(self.sp, h.vocab_size)
        self.serialized_grammar_tables = serialize_grammar_tables(self.grammar_tables)
        self.train_specs, self.train_total_tokens = build_shard_specs(h.train_files)
        self.val_specs, self.val_total_tokens = build_shard_specs(h.val_files)
        all_train_docs = build_doc_spans(self.train_specs, self.bos_id, h.train_doc_limit, h.train_token_limit, keep_last_open_doc=False)
        all_val_docs = build_doc_spans(self.val_specs, self.bos_id, h.val_doc_limit, h.val_token_limit, keep_last_open_doc=True)
        self.all_train_doc_spans = all_train_docs
        self.all_val_doc_spans = all_val_docs
        self.train_doc_spans = all_train_docs[h.rank::h.world_size]
        self.val_doc_spans = all_val_docs[h.rank::h.world_size]
        self.eval_loader = None
        h._grammar_tables = self.grammar_tables
        h._serialized_grammar_tables = self.serialized_grammar_tables
        h._pad_id = self.pad_id
        h._bos_id = self.bos_id
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
        device_tokens = h.train_batch_tokens // (h.world_size * h.grad_accum_steps)
        self.device_batch_size = device_tokens // h.train_seq_len
        if self.device_batch_size <= 0:
            raise ValueError("TRAIN_BATCH_TOKENS is too small for document-local loader")
        dataset = DocumentWindowDataset(
            h._train_specs,
            h._train_doc_spans,
            h.train_seq_len,
            mode="train",
            epoch=0,
            use_grammar=h.use_grammar,
            grammar_tables=h._serialized_grammar_tables if h.use_grammar else None,
            pad_id=h._pad_id,
            eval_stride=h.eval_stride,
            shuffle_docs=h.shuffle_docs,
            seed=h.seed + h.rank,
            feature_cache_docs=h.train_feature_cache_docs,
            cache_max_doc_tokens=h.cache_max_doc_tokens,
        )
        loader = make_loader(
            dataset,
            self.device_batch_size,
            h.num_workers,
            False,
            h.prefetch_factor,
            h.persistent_workers,
            True,
        )
        self.prefetcher = DevicePrefetcher(loader, device, h.pin_memory, h.host_prefetch_batches)

    def next_batch(self, global_tokens, grad_accum_steps):
        if self.fallback is not None:
            return self.fallback.next_batch(global_tokens, grad_accum_steps)
        batch = self.prefetcher.next()
        if batch is None:
            raise RuntimeError("Document-local training loader unexpectedly exhausted")
        return batch


class PatchedCausalSelfAttention(base.CausalSelfAttention):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.local_window = 0
        self.local_chunk = 0

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = base.apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = base.apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if self.local_window > 0 and self.local_window < seqlen:
            qh = q.permute(0, 2, 1, 3)
            kh = k.permute(0, 2, 1, 3)
            vh = v.permute(0, 2, 1, 3)
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                kh = kh.repeat_interleave(repeat, dim=1)
                vh = vh.repeat_interleave(repeat, dim=1)
            y = sliding_window_attention(qh, kh, vh, self.local_window, self.local_chunk, self.training).permute(0, 2, 1, 3)
        else:
            y = base.causal_attention(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


class PatchedBlock(base.Block):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len, layer_idx=0, ln_scale=False):
        super().__init__(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len, layer_idx=layer_idx, ln_scale=ln_scale)
        self.conv_norm = None
        self.conv = None
        self.conv_proj = None
        self.conv_kernel = 0

    def enable_local(self, dim: int, local_window: int, local_chunk: int, conv_kernel: int) -> None:
        self.attn.local_window = int(local_window)
        self.attn.local_chunk = int(local_chunk)
        self.conv_kernel = int(conv_kernel)
        if self.conv_kernel > 1:
            self.conv_norm = base.RMSNorm()
            self.conv = nn.Conv1d(dim, dim, kernel_size=self.conv_kernel, groups=dim, bias=False)
            self.conv_proj = base.CastedLinear(dim, dim, bias=False)
            self.conv_proj._zero_init = True
            nn.init.zeros_(self.conv_proj.weight)

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.conv is not None and self.conv_norm is not None and self.conv_proj is not None:
            conv_in = self.conv_norm(x_in).transpose(1, 2)
            conv_in = F.pad(conv_in, (self.conv_kernel - 1, 0))
            conv_out = self.conv(conv_in).transpose(1, 2)
            x_in = x_in + self.conv_proj(conv_out)
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)


class PatchedGPT(base.GPT):
    def __init__(self, h):
        super().__init__(h)
        self.grammar_adapter = GrammarStateEmbedding(h._grammar_tables, h.embedding_dim, init_scale=h.grammar_init_scale) if h.use_grammar else None
        if h.local_attn_layers > 0:
            for idx, block in enumerate(self.blocks):
                if idx < h.local_attn_layers:
                    block.enable_local(h.model_dim, h.local_window, h.local_attn_chunk, h.local_conv_kernel)

    def forward_logits(self, input_ids, grammar_feature_ids=None):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.grammar_adapter is not None:
            if grammar_feature_ids is not None and grammar_feature_ids.size(-1) > 0:
                x = x + self.grammar_adapter.embed_feature_ids(grammar_feature_ids).to(dtype=x.dtype)
            else:
                x = x + self.grammar_adapter(input_ids).to(dtype=x.dtype)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        for i in enc_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    gate = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, gate)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, grammar_feature_ids=None, loss_mask=None):
        logits = self.forward_logits(input_ids, grammar_feature_ids=grammar_feature_ids)
        if target_ids.dtype != torch.long:
            target_ids = target_ids.long()
        per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="none").view_as(target_ids)
        if loss_mask is None:
            return per_token.mean()
        mask = loss_mask.to(dtype=per_token.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_token * mask).sum() / denom


def build_eval_loader(h, val_data: ValidationData) -> DataLoader | None:
    if not h.doc_local_windows:
        return None
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    batch_size = max(local_batch_tokens // h.eval_seq_len, 1)
    dataset = DocumentWindowDataset(
        h._val_specs,
        h._val_doc_spans,
        h.eval_seq_len,
        mode="eval",
        epoch=0,
        use_grammar=h.use_grammar,
        grammar_tables=h._serialized_grammar_tables if h.use_grammar else None,
        pad_id=h._pad_id,
        eval_stride=h.eval_stride,
        shuffle_docs=False,
        seed=h.seed,
        feature_cache_docs=0,
        cache_max_doc_tokens=0,
    )
    return make_loader(dataset, batch_size, h.eval_num_workers, False, h.prefetch_factor, h.persistent_workers, False)


def _loss_bpb(loss_sum: Tensor, token_count: Tensor, byte_count: Tensor) -> tuple[float, float]:
    return base._loss_bpb(loss_sum, token_count, byte_count)


def evaluate_doc_loader(h, device: torch.device, val_data: ValidationData, model, loader: DataLoader | None) -> tuple[float, float]:
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL(h, device, val_data, model)
    assert loader is not None
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    prefetcher = DevicePrefetcher(loader, device, h.pin_memory, h.host_prefetch_batches)
    with torch.inference_mode():
        while True:
            batch = prefetcher.next()
            if batch is None:
                break
            x, y, feats, loss_mask = batch
            grammar_feature_ids = feats if feats.size(-1) > 0 else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, grammar_feature_ids=grammar_feature_ids, loss_mask=loss_mask).detach()
            batch_tokens = loss_mask.to(dtype=torch.float64).sum()
            loss_sum += batch_loss.to(torch.float64) * batch_tokens
            token_count += batch_tokens
            prev_ids = x.reshape(-1).long()
            tgt_ids = y.reshape(-1).long()
            flat_mask = loss_mask.reshape(-1).bool()
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes[flat_mask].to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, base.CastedLinear) and module.weight.numel() > 65536:
            cat = base.classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def output_hook(module, inp, out):
            x = out.detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if "tok_emb.weight" not in hessians:
                hessians["tok_emb.weight"] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians["tok_emb.weight"].addmm_(x.T, x)

        hooks.append(hook_module.register_forward_hook(output_hook))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            batch = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            if len(batch) == 2:
                x, _ = batch
                grammar_feature_ids = None
            else:
                x, _, feats, _ = batch
                grammar_feature_ids = feats if feats.size(-1) > 0 else None
            model.forward_logits(x, grammar_feature_ids=grammar_feature_ids)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    passthrough_prefixes = ("grammar_adapter.",)
    for name, tensor in state_dict.items():
        if name.startswith(passthrough_prefixes):
            t = tensor.detach().cpu().contiguous()
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
    if meta:
        filtered_state = {name: tensor for name, tensor in state_dict.items() if name not in meta}
        base_result, base_meta = ORIG_GPTQ_MIXED_QUANTIZE(filtered_state, hessians, h)
        result.update(base_result)
        meta.update(base_meta)
        return result, meta
    return ORIG_GPTQ_MIXED_QUANTIZE(state_dict, hessians, h)


def train_model(h, device: torch.device, val_data: ValidationData):
    base_model = base.GPT(h).to(device).bfloat16()
    base.restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    base.log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    optimizers = base.Optimizers(h, base_model)
    train_loader = base.ShuffledSequenceLoader(h, device)
    val_data.eval_loader = build_eval_loader(h, val_data)
    max_wallclock_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        base.log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            batch = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            if len(batch) == 2:
                x, y = batch
                feats = None
                loss_mask = None
            else:
                x, y, feats, loss_mask = batch
                feats = feats if feats.size(-1) > 0 else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, grammar_feature_ids=feats, loss_mask=loss_mask)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [base.copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                base.log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            base.log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    base.log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = base.ShuffledSequenceLoader(h, device)

    ema_state = {name: tensor.detach().float().clone() for name, tensor in base_model.state_dict().items()}
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(h, device, val_data, model)
            base.log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                base.log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            base.log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for name, tensor in base_model.state_dict().items():
                ema_state[name].mul_(h.ema_decay).add_(tensor.detach().float(), alpha=1.0 - h.ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            extra = ""
            if h.use_grammar and getattr(base_model, "grammar_adapter", None) is not None:
                extra = f" grammar_gate:{float(base_model.grammar_adapter.gate.detach().cpu()):.5f}"
            base.log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}{extra}")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    base.log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
    base.log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def eval_val(h, device, val_data, model):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL(h, device, val_data, model)
    if getattr(val_data, "eval_loader", None) is None:
        val_data.eval_loader = build_eval_loader(h, val_data)
    return evaluate_doc_loader(h, device, val_data, model, val_data.eval_loader)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL_SLIDING(h, device, val_data, base_model, batch_seqs=batch_seqs)
    if getattr(val_data, "eval_loader", None) is None:
        val_data.eval_loader = build_eval_loader(h, val_data)
    return evaluate_doc_loader(h, device, val_data, base_model, val_data.eval_loader)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    if not h.doc_local_windows:
        return ORIG_EVAL_VAL_TTT(h, device, val_data, base_model, batch_seqs=batch_seqs)
    raise NotImplementedError("Document-local TTT has not been ported yet; run with TTT_ENABLED=0 for this experiment.")


def submission_source_text() -> str:
    folder = Path(__file__).resolve().parent
    parts = []
    for path in (folder / "train_gpt.py", folder / "record_base.py", folder / "grammar_features.py"):
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
        base.log(f"doc_windows: local_train_seq={h.train_seq_len} local_eval_seq={h.eval_seq_len} eval_stride={h.eval_stride}")
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
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    base.timed_eval("quantized", base.eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        base.timed_eval("quantized_doc_rolling" if h.doc_local_windows else "quantized_sliding_window", base.eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        if h.doc_local_windows:
            base.log("ttt:skipped because document-local TTT is not yet ported in this experiment")
        else:
            del eval_model, compiled_model
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            ttt_model = base.deserialize(h, device)
            if h.num_loops > 0:
                ttt_model.looping_active = True
            base.timed_eval("quantized_ttt", base.eval_val_ttt, h, device, val_data, ttt_model)
            del ttt_model


base.ValidationData = ValidationData
base.ShuffledSequenceLoader = ShuffledSequenceLoader
base.CausalSelfAttention = PatchedCausalSelfAttention
base.Block = PatchedBlock
base.GPT = PatchedGPT
base.collect_hessians = collect_hessians
base.gptq_mixed_quantize = gptq_mixed_quantize
base.train_model = train_model
base.eval_val = eval_val
base.eval_val_sliding = eval_val_sliding
base.eval_val_ttt = eval_val_ttt
base.train_and_eval = train_and_eval


def main():
    base.main()


if __name__ == "__main__":
    main()
