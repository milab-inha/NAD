# NAD MRI Sampling

This repository contains the NAD-only sampling code separated from the original comparison workspace.

## Structure

- `scripts/image_sample.py`: NAD sampling entry point.
- `scripts/noise_estimators.py`: noise estimators used by NAD.
- `scripts/utils_hk.py`: reconstruction metrics and progress meters.
- `improved_diffusion/`: model, diffusion, distributed, and MRI data loading code required by the sampler.

## Installation

```bash
conda env create -f environment.yml
conda activate nad-mri
pip install -e .
```

If your machine uses a different CUDA version, adjust `pytorch-cuda` in `environment.yml`.
`meddlr` and dataset access may require the same environment used by the original project.

## Sampling

```bash
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
```

Outputs are written to:

```text
output/nad_mask_<mask_pattern>_acc_rate_<acc_rate>_steps_<steps>_noise_<noise_estimator>/
```

Noise estimator options:

- `pca` default
- `mri_wavelet`
- `tv`
- `block_adaptive_gaussian`
- `adaptive_wavelet`
- `scale_invariance`
