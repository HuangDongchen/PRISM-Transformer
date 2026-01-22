# PRISM (nanoGPT) — Public Release

This repository is a PRISM-modified fork of nanoGPT that adds:
- PRISM attention (`u_proj`, signal/noise split, differential denoising via `lambda_q`)
- TinyStories (GPT-2 BPE) data pipeline
- TensorBoard + metrics logging
- Automated visualization scripts
- Paper appearing on arXiv soon
## One‑Click Run

From the repo root:

```bash
python run_experiment.py
```

`run_experiment.py` orchestrates:
1) TinyStories data preparation (`data/tinystories/prepare.py`)
2) Training (`train.py config/train_prism_tiny.py`)
3) Visualization (`analysis/viz_prism.py`)

## Requirements

### Required (manual)
- **PyTorch**: install a suitable build for your platform (CUDA recommended).

### Auto-installed by `run_experiment.py`
If missing, the script will automatically install:
- `tiktoken`, `datasets`, `tqdm`, `matplotlib`, `tensorboard`

To disable auto-install:
```bash
python run_experiment.py --no_install_deps
```

## License

This public sub-repository is distributed under the Apache License 2.0.

It also includes code derived from upstream `nanoGPT` (MIT licensed); see `NOTICE` and
`THIRD_PARTY_LICENSES/` for attribution and the upstream license text.

## Notes

- The TinyStories download is large and may take time and disk space.
- On Windows, `torch.compile` may not be supported; `train.py` handles this and will fall back to non-compiled execution.
