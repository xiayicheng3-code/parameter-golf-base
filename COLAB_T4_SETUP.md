# Google Colab T4 x1 Setup

This is a standardized, beginner-friendly procedure for getting a usable Parameter Golf environment on a single Google Colab T4.

Use this for:

- environment checks
- package setup without guessing what Colab already has
- a safe first smoke run
- packaging the log and model artifacts for download

Do not use this as evidence for leaderboard timing. Colab T4 is useful for smoke tests and debugging, not for the official 8xH100 constraint.

## 0. Switch Colab to GPU

In Colab:

1. `Runtime`
2. `Change runtime type`
3. Hardware accelerator: `T4 GPU`
4. Save

If you already had a notebook open on CPU, it is often cleaner to restart the runtime once after switching.

## 1. Probe the environment first

Run this Python cell before installing anything:

```python
import importlib
import platform
import subprocess
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
print()
print("GPU info:")
subprocess.run(["nvidia-smi"], check=False)
print()

modules = [
    "torch",
    "numpy",
    "sentencepiece",
    "datasets",
    "huggingface_hub",
    "tiktoken",
    "tqdm",
]

for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK   version={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: MISSING/BROKEN   {exc}")
```

What you want to see:

- `nvidia-smi` shows one T4
- `torch` imports successfully
- `torch` can see CUDA in the next step

## 2. Keep Colab's PyTorch if it already works

Do not blindly reinstall `torch` on Colab unless it is actually broken. Replacing Colab's preinstalled CUDA-matched PyTorch is a common way to create setup problems.

Run this cell:

```python
import importlib
import subprocess
import sys

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch is not usable in this runtime. Restart the Colab GPU runtime first, "
        "then re-run the notebook before trying manual torch installation."
    ) from exc

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda toolkit reported by torch:", torch.version.cuda)

required = {
    "sentencepiece": "sentencepiece",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "tiktoken": "tiktoken",
    "tqdm": "tqdm",
}

missing = []
for module_name, package_name in required.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)

if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("No extra package install needed.")
```

## 3. Clone the repo into Colab

Run:

```bash
%cd /content
!rm -rf /content/parameter-golf-base
!git clone https://github.com/xiayicheng3-code/parameter-golf-base.git
%cd /content/parameter-golf-base
```

If you want your own branch or local edits later, you can clone your fork instead.

## 4. Download a small cached dataset subset

Start with one training shard for a smoke test:

```bash
%cd /content/parameter-golf-base
!python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This is much more realistic than a toy run, but still smaller than the full training set.

## 5. Run a safe first smoke test on 1xT4

Use a conservative batch size first. This is meant to be stable, not fast.

```bash
%%bash
cd /content/parameter-golf-base

RUN_ID=colab_t4_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 \
TRAIN_LOG_EVERY=25 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee colab_t4_smoke.log
```

Why these settings:

- `TRAIN_BATCH_TOKENS=8192` is intentionally small for a first T4 run
- `VAL_BATCH_SIZE=8192` keeps validation memory pressure low too
- `TRAIN_LOG_EVERY=25` gives readable logs
- `MAX_WALLCLOCK_SECONDS=120` makes the first check short

If you get CUDA OOM, try this smaller variant:

```bash
%%bash
cd /content/parameter-golf-base

RUN_ID=colab_t4_smoke_small \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
TRAIN_BATCH_TOKENS=4096 \
VAL_BATCH_SIZE=4096 \
TRAIN_LOG_EVERY=25 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee colab_t4_smoke_small.log
```

## 6. Package the log and model artifacts

The baseline script writes:

- `final_model.pt`
- `final_model.int8.ptz`

Package them with the log:

```bash
%%bash
cd /content/parameter-golf-base

mkdir -p colab_artifacts
cp colab_t4_smoke.log colab_artifacts/
cp final_model.pt colab_artifacts/
cp final_model.int8.ptz colab_artifacts/
tar -czf colab_t4_smoke_artifacts.tar.gz -C colab_artifacts .
ls -lh colab_artifacts colab_t4_smoke_artifacts.tar.gz
```

Download to your laptop with a normal Colab download:

```python
from google.colab import files
files.download("/content/parameter-golf-base/colab_t4_smoke_artifacts.tar.gz")
```

## 7. A simple rule for future Colab sessions

When you open a fresh Colab runtime, do these in order:

1. Confirm GPU with `nvidia-smi`
2. Confirm `torch.cuda.is_available()`
3. Install only missing non-torch packages
4. Clone repo
5. Download `--train-shards 1`
6. Run the conservative smoke command
7. Only after that, increase batch sizes or wallclock

## 8. If you want a more T4-oriented experiment

There is already a notebook-friendly T4/Kaggle-oriented script here:

- [experiments/m34_multi_hash_gated_kaggle_t4/train_gpt.py](/Users/yichengxia/ML_NN_DA/parameter_golf/parameter-golf-base/experiments/m34_multi_hash_gated_kaggle_t4/train_gpt.py)

That script is more specialized than the baseline:

- it auto-downloads cached assets if missing
- it is designed to be edited directly in the file instead of relying on many environment variables
- it has T4-aware dtype and attention fallbacks

For your first Colab pass, I still recommend using the baseline smoke procedure above because it is easier to debug.
