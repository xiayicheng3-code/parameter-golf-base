from __future__ import annotations

import argparse
import io
import lzma
from pathlib import Path

import torch


def selective_prune_saved_int6(
    input_path: Path,
    output_path: Path,
    code_path: Path,
    target_mb: float,
) -> None:
    payload = torch.load(
        io.BytesIO(lzma.decompress(input_path.read_bytes())),
        map_location="cpu",
    )
    quant_result: dict[str, torch.Tensor] = payload["w"]
    quant_meta: dict[str, object] = payload["m"]

    code_bytes = len(code_path.read_text(encoding="utf-8").encode("utf-8"))
    target_bytes = int(target_mb * 1024 * 1024)

    ones_info: list[tuple[str, int, float]] = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim == 0:
            continue
        ones_mask = q.abs() == 1
        if not ones_mask.any():
            continue
        row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
        flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
        errors = s.float()[row_idx].pow(2)
        for fi, err in zip(flat_idx.tolist(), errors.tolist()):
            ones_info.append((qk, fi, err))

    ones_info.sort(key=lambda x: x[2])

    def try_prune(n: int) -> tuple[int, dict[str, torch.Tensor]]:
        tmp = {k: v.clone() for k, v in quant_result.items()}
        for i in range(min(n, len(ones_info))):
            tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
        buf = io.BytesIO()
        torch.save({"w": tmp, "m": quant_meta}, buf)
        quant_blob = lzma.compress(buf.getvalue(), preset=9)
        return len(quant_blob) + code_bytes, tmp

    unpruned_total_bytes, _ = try_prune(0)
    print(
        f"selective_prune_saved: {len(ones_info)} ±1 candidates, "
        f"unpruned={(unpruned_total_bytes / (1024 * 1024)):.2f}MB target={target_mb}MB"
    )

    if not ones_info:
        print("selective_prune_saved: no ±1 candidates, copying input artifact")
        output_path.write_bytes(input_path.read_bytes())
        return

    if unpruned_total_bytes <= target_bytes:
        print("selective_prune_saved: already fits, no pruning needed")
        output_path.write_bytes(input_path.read_bytes())
        return

    fully_pruned_total_bytes, _ = try_prune(len(ones_info))
    print(f"selective_prune_saved: full ±1 prune={(fully_pruned_total_bytes / (1024 * 1024)):.2f}MB")

    if fully_pruned_total_bytes > target_bytes:
        prune_n = len(ones_info)
    else:
        lo, hi = 0, len(ones_info)
        while lo < hi:
            mid = (lo + hi) // 2
            total_bytes, _ = try_prune(mid)
            if total_bytes <= target_bytes:
                hi = mid
            else:
                lo = mid + 1
        prune_n = lo

    _, pruned = try_prune(prune_n)
    buf = io.BytesIO()
    torch.save({"w": pruned, "m": quant_meta}, buf)
    quant_blob = lzma.compress(buf.getvalue(), preset=9)
    output_path.write_bytes(quant_blob)
    print(
        f"selective_prune_saved: pruning {prune_n}/{len(ones_info)} ±1 values "
        f"({100.0 * prune_n / max(len(ones_info), 1):.1f}%)"
    )
    print(f"selective_prune_saved: model_bytes={len(quant_blob)} total_bytes={len(quant_blob) + code_bytes}")
    print(f"selective_prune_saved: wrote {output_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Re-run selective ±1 pruning on a saved int6 artifact.")
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "final_model.int6.ptz",
        help="Path to the saved int6 artifact (default: ./final_model.int6.ptz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "final_model.pruned.int6.ptz",
        help="Path to write the pruned artifact (default: ./final_model.pruned.int6.ptz)",
    )
    parser.add_argument(
        "--code",
        type=Path,
        default=here / "train_gpt.py",
        help="Path to the train_gpt.py used for code-byte accounting",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=15.24,
        help="Target total size in MiB-style accounting, matching train_gpt.py",
    )
    args = parser.parse_args()
    selective_prune_saved_int6(args.input, args.output, args.code, args.target_mb)


if __name__ == "__main__":
    main()
