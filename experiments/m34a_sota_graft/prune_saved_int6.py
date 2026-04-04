from __future__ import annotations

import argparse
import os
from pathlib import Path

from train_gpt import rerun_selective_prune_saved_artifact


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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of worker threads for parallel probe evaluation",
    )
    args = parser.parse_args()
    rerun_selective_prune_saved_artifact(
        input_path=args.input,
        output_path=args.output,
        code_path=args.code,
        target_mb=args.target_mb,
        workers=max(args.workers, 1),
        log_fn=print,
    )


if __name__ == "__main__":
    main()
