#!/usr/bin/env bash
# Training di Uformer sul dataset DIV2K degradato.
# Va lanciato dalla radice del repository.
set -e

python -m src.training.train_uformer \
    --arch Uformer_B --batch_size 32 --gpu '0,1' \
    --train_ps 128 --train_dir data/DIV2K_train_degraded \
    --val_dir data/DIV2K_valid_degraded \
    --save_dir experiments/logs/ --env _denoise \
    --dataset div2k_custom --warmup
