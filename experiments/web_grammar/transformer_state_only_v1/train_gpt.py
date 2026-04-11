from __future__ import annotations

from dataclasses import dataclass
import glob
import io
import json
import math
import os
from pathlib import Path
import random
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from grammar_features import GrammarStateEmbedding, build_vocab_tables_from_sentencepiece


ROOT = Path(__file__).resolve().parents[3]


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
    run_id: str = os.environ.get("RUN_ID", "state_only_grammar_smoke")
    data_path: str = os.environ.get("DATA_PATH", str(ROOT / "data/datasets/fineweb10B_sp1024"))
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", str(ROOT / "data/tokenizers/fineweb_1024_bpe.model"))
    vocab_size: int = env_int("VOCAB_SIZE", 1024)
    train_token_limit: int = env_int("TRAIN_TOKEN_LIMIT", 200_000)
    val_token_limit: int = env_int("VAL_TOKEN_LIMIT", 50_000)
    seq_len: int = env_int("SEQ_LEN", 128)
    batch_size: int = env_int("BATCH_SIZE", 8)
    steps: int = env_int("STEPS", 50)
    eval_every: int = env_int("EVAL_EVERY", 25)
    d_model: int = env_int("D_MODEL", 160)
    n_layers: int = env_int("N_LAYERS", 4)
    n_heads: int = env_int("N_HEADS", 4)
    dropout: float = env_float("DROPOUT", 0.0)
    lr: float = env_float("LR", 3e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 0.01)
    seed: int = env_int("SEED", 1337)
    use_grammar: bool = env_bool("USE_GRAMMAR", True)
    grammar_init_scale: float = env_float("GRAMMAR_INIT_SCALE", 0.02)
    device: str = os.environ.get("DEVICE", "auto")

    @property
    def train_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_train_*.bin")

    @property
    def val_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_val_*.bin")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    return torch.from_numpy(tokens_np.astype(np.int64, copy=False))


def load_token_split(pattern: str, limit: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    chunks: list[Tensor] = []
    remaining = limit if limit > 0 else None
    for file in files:
        shard = load_data_shard(file)
        if remaining is not None:
            if remaining <= 0:
                break
            shard = shard[:remaining]
            remaining -= int(shard.numel())
        chunks.append(shard)
        if remaining == 0:
            break
    tokens = torch.cat(chunks).contiguous()
    if tokens.numel() < 2:
        raise ValueError(f"Need at least two tokens for {pattern}")
    return tokens


def take_cyclic(tokens: Tensor, start: int, count: int) -> Tensor:
    if count > tokens.numel():
        repeats = math.ceil(count / tokens.numel()) + 1
        tokens = tokens.repeat(repeats)
        start %= tokens.numel()
    end = start + count
    if end <= tokens.numel():
        return tokens[start:end]
    return torch.cat((tokens[start:], tokens[: end - tokens.numel()]))


def make_batch(tokens: Tensor, batch_size: int, seq_len: int, step: int, device: torch.device) -> tuple[Tensor, Tensor]:
    span = batch_size * seq_len + 1
    start = (step * batch_size * seq_len) % max(tokens.numel() - 1, 1)
    local = take_cyclic(tokens, start, span).to(device=device, dtype=torch.long)
    return local[:-1].reshape(batch_size, seq_len), local[1:].reshape(batch_size, seq_len)


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
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("D_MODEL must be divisible by N_HEADS")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, dim = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, steps, dim)
        x = x + self.dropout(self.proj(y))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class StateOnlyTransformer(nn.Module):
    def __init__(self, cfg: Config, tables) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.grammar_adapter = (
            GrammarStateEmbedding(tables, cfg.d_model, init_scale=cfg.grammar_init_scale)
            if cfg.use_grammar
            else None
        )
        self.blocks = nn.ModuleList([CausalBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
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

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        _, steps = input_ids.shape
        positions = torch.arange(steps, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        if self.grammar_adapter is not None:
            x = x + self.grammar_adapter(input_ids).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        if target_ids is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokens: Tensor,
    cfg: Config,
    device: torch.device,
    base_bytes: Tensor,
    has_leading_space: Tensor,
    is_boundary: Tensor,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    seqs = max((tokens.numel() - 1) // cfg.seq_len, 1)
    eval_batch = max(1, cfg.batch_size)
    for seq_start in range(0, seqs, eval_batch):
        seq_end = min(seq_start + eval_batch, seqs)
        raw = tokens[seq_start * cfg.seq_len : seq_end * cfg.seq_len + 1].to(device=device, dtype=torch.long)
        batch = seq_end - seq_start
        x = raw[:-1].reshape(batch, cfg.seq_len)
        y = raw[1:].reshape(batch, cfg.seq_len)
        loss = model(x, y)
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = base_bytes[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space[tgt_ids] & ~is_boundary[prev_ids]).to(dtype=torch.int16)
        total_bytes += float(token_bytes.to(torch.float64).sum().item())
    model.train()
    val_loss = total_loss / max(total_tokens, 1)
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
    tables = build_vocab_tables_from_sentencepiece(sp, cfg.vocab_size)
    train_tokens = load_token_split(cfg.train_files, cfg.train_token_limit)
    val_tokens = load_token_split(cfg.val_files, cfg.val_token_limit)
    base_bytes, has_leading_space, is_boundary = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    model = StateOnlyTransformer(cfg, tables).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path.write_text("", encoding="utf-8")
    log(json.dumps({**cfg.__dict__, "device_resolved": str(device), "params": sum(p.numel() for p in model.parameters())}, indent=2))
    log(f"train_tokens:{train_tokens.numel()} val_tokens:{val_tokens.numel()} grammar_enabled:{cfg.use_grammar}")

    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        x, y = make_batch(train_tokens, cfg.batch_size, cfg.seq_len, step, device)
        loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step <= 5 or step == cfg.steps or (cfg.eval_every > 0 and step % cfg.eval_every == 0):
            gate = float(model.grammar_adapter.gate.detach().cpu()) if model.grammar_adapter is not None else 0.0
            log(f"step:{step}/{cfg.steps} train_loss:{loss.item():.4f} grammar_gate:{gate:.6f}")
        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step != cfg.steps:
            val_loss, val_bpb = evaluate(model, val_tokens, cfg, device, base_bytes, has_leading_space, is_boundary)
            log(f"step:{step}/{cfg.steps} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")

    val_loss, val_bpb = evaluate(model, val_tokens, cfg, device, base_bytes, has_leading_space, is_boundary)
    artifact_bytes, code_bytes, total_bytes, artifact_path = export_compressed_model(model, folder, cfg.run_id)
    elapsed = time.perf_counter() - t0
    log(f"final val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")
    log(f"artifact_bytes:{artifact_bytes} code_bytes:{code_bytes} total_submission_bytes:{total_bytes}")
    log(f"artifact_path:{artifact_path}")
    log(f"elapsed_seconds:{elapsed:.3f} log_path:{log_path}")


if __name__ == "__main__":
    main()
