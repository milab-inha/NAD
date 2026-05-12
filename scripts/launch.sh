#!/usr/bin/env bash
set -euo pipefail

python scripts/image_sample.py \
  --model_nad_path /path/to/model.pt \
  --data_path /path/to/skm-tea \
  --output_dir output \
  --steps 50 \
  --noise_estimator pca \
  --batch_size 4 \
  --mask_pattern random \
  --acc_rate 4 \
  --which_gpu 0
