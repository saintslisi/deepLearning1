#!/usr/bin/env bash
# Valutazione di Uformer sul validation set DIV2K degradato.
# Va lanciato dalla radice del repository.
set -e

WEIGHTS=${1:-experiments/logs/denoising/div2k_custom/Uformer_B_/models/model_best.pth}

python -m src.evaluation.test_uformer_full_frame \
    --input_dir data/DIV2K_valid_degraded \
    --gt_dir data/DIV2K_valid_HR \
    --result_dir results/uformer \
    --weights "$WEIGHTS"
