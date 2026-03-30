# M34 Kaggle T4 Fork

This is a Kaggle/T4-oriented copy of `m34_multi_hash_gated`.

Goals of this fork:

- keep the original `m34` baseline untouched
- avoid training/config environment variables
- make the script notebook-friendly: edit the config block inside `main()`
- support T4 by falling back from `flash_attn_interface` to PyTorch SDPA
- prefer `float16` automatically on pre-Ampere GPUs

Current status:

- `train_gpt.py` is the only file intended for real use here
- `NGRAM_CANDIDATE_SOURCE` should still stay `inline`
- the Triton prototype remains research-only and is not wired into training
- if the cached FineWeb shards or tokenizer are missing, the script downloads them directly from Hugging Face into a local `data/` folder next to the script

Important differences from the main M34 script:

- configuration is hardcoded in `main()`
- multi-GPU launch uses `torch.multiprocessing.spawn` instead of `torchrun` env vars
- the script auto-detects `1` vs `2` visible GPUs
- default batching is aimed at Kaggle `2xT4`

Notebook usage:

1. Open [train_gpt.py](/Users/yichengxia/ML_NN_DA/parameter_golf/parameter-golf-base/experiments/m34_multi_hash_gated_kaggle_t4/train_gpt.py)
2. Edit the config block near the bottom inside `main()`
3. The first run will auto-download the selected tokenizer variant and shard prefix if they are missing
3. Run it from a notebook cell with:

```bash
!cd /kaggle/working/m34_multi_hash_gated_kaggle_t4 && python train_gpt.py
```

Recommended first checks on Kaggle:

- confirm the dataset/tokenizer paths in `main()`
- start with `use_compile=True`
- keep `NGRAM_CANDIDATE_SOURCE="inline"`
- if compile is flaky on a given image, flip `args.use_compile = False` in `main()`
