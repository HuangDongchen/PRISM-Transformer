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

### The $\pi$-Trick: Irrational Frequency Scaling

Looking for a stable RoPE configuration for long-context or deep models? 
Based on our theoretical findings from the **KAM Theorem** (Kolmogorov-Arnold-Moser), we suspect that **irrational frequencies** are essential to prevent spectral resonance between semantic signals and syntactic noise.

We invite the community to try the **$\pi$-RoPE** patch. It's a one-line change:

```python
# In your rotary_embedding.py
# Standard RoPE
# inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

# The Pi-Trick (PRISM)
import math
inv_freq = 1.0 / ((base * math.pi) ** (torch.arange(0, dim, 2).float() / dim))
```

### The Golden Ratio Challenge ($\phi$-RoPE)
Theory suggests that the Golden Ratio ($\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$) is the "most irrational number" (hardest to approximate by rationals). While we used $\pi$ for better spectral separation, $\phi$ might offer superior resonance protection in certain regimes.

```python
# The Phi-Trick
phi = (1 + 5 ** 0.5) / 2
inv_freq = 1.0 / ((base * phi) ** (torch.arange(0, dim, 2).float() / dim))
```

## Does $\pi$ win? Or does $\phi$ rule them all?





## License

This public sub-repository is distributed under the Apache License 2.0.

It also includes code derived from upstream `nanoGPT` (MIT licensed); see `NOTICE` and
`THIRD_PARTY_LICENSES/` for attribution and the upstream license text.

## Notes

- The TinyStories download is large and may take time and disk space.
- On Windows, `torch.compile` may not be supported; `train.py` handles this and will fall back to non-compiled execution.
