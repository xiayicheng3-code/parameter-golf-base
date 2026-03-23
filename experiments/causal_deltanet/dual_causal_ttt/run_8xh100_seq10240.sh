#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RUN_ID="${RUN_ID:-shared_delta_slide_seq10240_v0}"
export DATA_PATH="${DATA_PATH:-../../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-../../../data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-10240}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-655360}"
export ATTENTION_IMPL="${ATTENTION_IMPL:-sliding_gqa}"
export SLIDING_WINDOW_SIZE="${SLIDING_WINDOW_SIZE:-512}"
export SLIDING_CHUNK_SIZE="${SLIDING_CHUNK_SIZE:-512}"
export USE_SHARED_DELTA="${USE_SHARED_DELTA:-1}"
export DELTA_USE_QK_NORM="${DELTA_USE_QK_NORM:-1}"
export DELTA_USE_OUTPUT_PROJ="${DELTA_USE_OUTPUT_PROJ:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

LOG_FILE="${LOG_FILE:-train.log}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py | tee "${LOG_FILE}"
