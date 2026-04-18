from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm

HEADER_BYTES = 256 * np.dtype("<i4").itemsize
TRAIN_SEQ_LEN = 2048
VOCAB_SIZE = 8192
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
TOKENIZER_PATH = DATA_DIR / "tokenizers" / f"fineweb_{VOCAB_SIZE}_bpe.model"
TRAIN_GLOB = str(DATA_DIR / "datasets" / f"fineweb10B_sp{VOCAB_SIZE}" / "fineweb_train_*.bin")


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
    return specs, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", default=TRAIN_GLOB)
    parser.add_argument(
        "--shards",
        default="",
        help="Comma-separated train shard ids to include, e.g. 32,64,96. Empty means all matched shards.",
    )
    return parser.parse_args()


def filter_specs_by_shard_ids(specs: list[ShardSpec], shard_ids: set[int]) -> list[ShardSpec]:
    if not shard_ids:
        return specs
    kept_paths = {
        f"fineweb_train_{shard_id:06d}.bin"
        for shard_id in shard_ids
    }
    kept_specs = [spec for spec in specs if Path(spec.path).name in kept_paths]
    if not kept_specs:
        raise FileNotFoundError(f"No requested shards found locally: {sorted(shard_ids)}")
    total = 0
    rebased: list[ShardSpec] = []
    for spec in kept_specs:
        rebased.append(ShardSpec(spec.path, total, spec.num_tokens))
        total += spec.num_tokens
    return rebased


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


def build_doc_spans(specs: list[ShardSpec], bos_id: int) -> list[DocSpan]:
    bos_positions = scan_bos_positions(specs, bos_id)
    total_tokens = specs[-1].global_start + specs[-1].num_tokens
    spans: list[DocSpan] = []
    for doc_id, start_pos in enumerate(bos_positions.tolist()):
        if doc_id + 1 < len(bos_positions):
            stop = min(int(bos_positions[doc_id + 1]) + 1, total_tokens)
            has_closing_bos = True
        else:
            break
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


def main() -> None:
    args = parse_args()
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer must define BOS")
    specs, _ = build_shard_specs(args.train_glob)
    shard_ids = {int(part) for part in args.shards.split(",") if part.strip()}
    specs = filter_specs_by_shard_ids(specs, shard_ids)
    total_tokens = specs[-1].global_start + specs[-1].num_tokens
    spans = build_doc_spans(specs, bos_id)
    seed = 42
    seq_len = TRAIN_SEQ_LEN

    doc_count = len(spans)
    pred_lens = np.asarray([span.pred_len for span in spans], dtype=np.int64)
    short_mask = pred_lens < seq_len
    short_docs = int(short_mask.sum())

    total_windows = 0
    total_pad = 0
    short_windows = 0
    short_pad = 0
    long_tail_windows = 0
    long_tail_pad = 0

    for span in spans:
        for epoch in (0, 1):
            for start in train_window_starts(span.pred_len, seq_len, span.doc_id, seed, epoch):
                valid_len = min(seq_len, span.pred_len - start)
                pad = seq_len - valid_len
                total_windows += 1
                total_pad += pad
                if span.pred_len < seq_len:
                    short_windows += 1
                    short_pad += pad
                elif pad > 0:
                    long_tail_windows += 1
                    long_tail_pad += pad

    unique_doc_slots = len(spans) * 2
    unique_capacity = unique_doc_slots * seq_len
    unique_pred_tokens = int(pred_lens.sum()) * 2
    unique_pad = unique_capacity - unique_pred_tokens

    print(f"train_glob={args.train_glob}")
    print(f"selected_shards={sorted(shard_ids) if shard_ids else 'all'}")
    print(f"available_train_shards={len(specs)} total_tokens={total_tokens}")
    print(f"doc_count={doc_count}")
    print(f"short_docs_lt_{seq_len}={short_docs} ({short_docs / max(doc_count,1):.2%})")
    print(f"pred_len_mean={pred_lens.mean():.2f} pred_len_p50={np.percentile(pred_lens,50):.0f} pred_len_p90={np.percentile(pred_lens,90):.0f} pred_len_p99={np.percentile(pred_lens,99):.0f}")
    print(f"pred_len_lt_256={(pred_lens < 256).sum()} ({((pred_lens < 256).sum()/max(doc_count,1)):.2%})")
    print(f"pred_len_lt_512={(pred_lens < 512).sum()} ({((pred_lens < 512).sum()/max(doc_count,1)):.2%})")
    print(f"pred_len_lt_1024={(pred_lens < 1024).sum()} ({((pred_lens < 1024).sum()/max(doc_count,1)):.2%})")
    print(f"pred_len_lt_1536={(pred_lens < 1536).sum()} ({((pred_lens < 1536).sum()/max(doc_count,1)):.2%})")
    print(f"epoch0+1_window_count={total_windows}")
    print(f"epoch0+1_pad_tokens={total_pad} ({total_pad / max(total_windows * seq_len,1):.2%} of slot capacity)")
    print(f"avg_pad_per_window={total_pad / max(total_windows,1):.2f}")
    print(f"short_doc_window_count={short_windows}")
    print(f"short_doc_pad_tokens={short_pad} ({short_pad / max(total_pad,1):.2%} of all pad)")
    print(f"avg_short_doc_pad_per_window={short_pad / max(short_windows,1):.2f}")
    print(f"long_doc_tail_window_count={long_tail_windows}")
    print(f"long_doc_tail_pad_tokens={long_tail_pad} ({long_tail_pad / max(total_pad,1):.2%} of all pad)")
    print(f"avg_long_tail_pad_per_window={long_tail_pad / max(long_tail_windows,1):.2f}")
    print(f"per_doc_epoch0or1_capacity={unique_capacity}")
    print(f"per_doc_epoch0or1_effective_pred_tokens={unique_pred_tokens}")
    print(f"per_doc_epoch0or1_pad_tokens={unique_pad} ({unique_pad / max(unique_capacity,1):.2%} of slot capacity)")


if __name__ == "__main__":
    main()
