# Deployment1Tx

Code for the paper:

> **Learning Coverage- and Power-Optimal Transmitter Placement from Building Maps: A Comparative Study of Direct and Indirect Neural Approaches**  
> Çağkan Yapar, TU Berlin

## Overview

This repository provides training and evaluation code for single-transmitter placement using two learning formulations:

- **Heatmap-based (indirect):** Predicts a received-power radio map from which the transmitter location is recovered via argmax. Includes discriminative models (DeepXL, PMNet, SIP2Net, DC-Net) and a diffusion model with multi-sample inference.
- **Score-map-based (direct):** Predicts a per-pixel optimality score over feasible transmitter locations. Supports single-objective argmax, top-K shortlisting with SAIPP-Net evaluation, and dual-objective strategies (minimax ranking, union pooling).

## Dataset

**RadioMapSeer-Deployment** — 167,525 urban building scenarios with dual ground-truth labels (coverage-optimal and power-optimal transmitter locations), obtained by exhaustive per-pixel SAIPP-Net evaluation.

Available at: [IEEE DataPort](https://dx.doi.org/10.21227/wjwa-th03)

### Expected data layout

```
data/
├── buildings/           # 256×256 building maps (from RadioMapSeer)
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── unified_results/     # RadioMapSeer-Deployment outputs
    ├── best_by_power/   # Power-optimal radio maps + Tx locations
    ├── best_by_coverage/# Coverage-optimal radio maps + Tx locations
    └── normalized_per_map/ # Score maps (16-bit PNG + metadata)
```

## Repository Structure

```
Deployment1Tx/
├── models/
│   ├── saipp_net.py          # SAIPP-Net (RadioNet) propagation model
│   ├── discriminative.py     # DeepXL, PMNet, SIP2Net, DC-Net (150×150)
│   └── diffusion.py          # DDPM with concat conditioning
│
├── data/
│   ├── dataset_heatmap.py    # Dataset for heatmap training (radio maps)
│   └── dataset_scoremap.py   # Dataset for score map training (16-bit)
│
├── training/
│   ├── train_heatmap.py      # Train discriminative heatmap models
│   ├── train_diffusion_heatmap.py  # Train diffusion heatmap models
│   ├── train_scoremap.py     # Train discriminative score map models
│   ├── train_diffusion_scoremap.py # Train diffusion score map models
│   └── losses.py             # Loss functions (L2, MS-SSIM, TV, GDL, etc.)
│
├── evaluation/
│   ├── eval_utils.py                    # Shared evaluation utilities
│   ├── eval_heatmap_utils.py            # Heatmap evaluation helpers
│   ├── eval_heatmap_discriminative.py   # Tables 3, 5
│   ├── eval_heatmap_diffusion_multisample.py  # Tables 2, 4
│   ├── eval_heatmap_diffusion_l2.py     # Table 6
│   ├── eval_scoremap_discriminative.py  # Tables 7–10 (discriminative)
│   ├── eval_scoremap_diffusion.py       # Tables 7–10 (diffusion)
│   ├── eval_minimax_discriminative.py   # Table 11 (discriminative)
│   ├── eval_minimax_diffusion.py        # Table 11 (diffusion)
│   └── eval_union.py                   # Table 12
│
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

```
torch >= 1.12
torchvision
numpy
Pillow
tqdm
```

Install: `pip install -r requirements.txt`

### SAIPP-Net Checkpoint

A pre-trained SAIPP-Net checkpoint is required for evaluation (SAIPP-Net evaluation of candidate placements). Place it at `checkpoints/saipp_net.pt` or specify via `--radionet_path`.

## Training

### Heatmap models (indirect)

**Discriminative** (e.g., DC-Net, power objective, DA-cGAN Stage 2 loss):
```bash
python -m training.train_heatmap \
    --arch dcnet_150 \
    --optimize_for power \
    --heatmap_type radio_power \
    --loss_ms_ssim_weight 8.4e6 --loss_tv_weight 1e3 --loss_focal_weight 1e7 \
    --buildings_dir data/buildings \
    --heatmap_base_dir data/unified_results \
    --use_center --augment \
    --epochs 200 --batch_size 16
```

**Diffusion** (e.g., power objective):
```bash
python -m training.train_diffusion_heatmap \
    --conditioning concat \
    --optimize_for power \
    --heatmap_type radio_power \
    --buildings_dir data/buildings \
    --heatmap_base_dir data/unified_results \
    --use_center --augment
```

### Score map models (direct)

**Discriminative** (e.g., PMNet, power score map):
```bash
python -m training.train_scoremap \
    --arch pmnet_150 \
    --optimize_for power \
    --heatmap_type avg_power_per_map \
    --buildings_dir data/buildings \
    --heatmap_base_dir data/unified_results \
    --use_center --augment
```

## Evaluation

### Heatmap models

```bash
# Discriminative (Tables 3, 5)
python -m evaluation.eval_heatmap_discriminative \
    --checkpoint_path path/to/model.pt \
    --optimize_for power --arch dcnet_150

# Diffusion multi-sample (Tables 2, 4, 6)
python -m evaluation.eval_heatmap_diffusion_multisample \
    --checkpoint_path path/to/diffusion.pt \
    --optimize_for power --num_samples 10
```

### Score map models

```bash
# Single-objective (Tables 7–10)
python -m evaluation.eval_scoremap_discriminative \
    --power_checkpoint path/to/power_scoremap.pt \
    --coverage_checkpoint path/to/cov_scoremap.pt \
    --k_values 1 200

# Minimax dual score map (Table 11)
python -m evaluation.eval_minimax_discriminative \
    --power_checkpoint path/to/power_scoremap.pt \
    --coverage_checkpoint path/to/cov_scoremap.pt \
    --k_values 200 500 1000 2000

# Union dual score map (Table 12)
python -m evaluation.eval_union \
    --power_checkpoint path/to/power_scoremap.pt \
    --coverage_checkpoint path/to/cov_scoremap.pt \
    --m_values 500 1000 2000
```

## Paper → Code Mapping

| Paper Table | Script | Key arguments |
|---|---|---|
| Table 2 (Diffusion, power) | `eval_heatmap_diffusion_multisample.py` | `--optimize_for power` |
| Table 3 (Discriminative, power) | `eval_heatmap_discriminative.py` | `--optimize_for power` |
| Table 4 (Diffusion, coverage) | `eval_heatmap_diffusion_multisample.py` | `--optimize_for coverage` |
| Table 5 (Discriminative, coverage) | `eval_heatmap_discriminative.py` | `--optimize_for coverage` |
| Table 6 (Diffusion, best-L2) | `eval_heatmap_diffusion_l2.py` | `--num_samples 10` |
| Tables 7–8 (Score map, single) | `eval_scoremap_discriminative.py` | `--k_values 1 200` |
| Tables 9–10 (Score map, best-L2) | `eval_scoremap_discriminative.py` | `--k_values 500 1000 2000` |
| Table 11 (Minimax) | `eval_minimax_discriminative.py` | `--k_values 200 500 1000 2000` |
| Table 12 (Union) | `eval_union.py` | `--m_values 500 1000 2000` |

## Architectures

| Paper name | `--arch` | Parameters |
|---|---|---|
| DeepXL | `deepxl_150` | ~20M |
| PMNet | `pmnet_150` | ~28M |
| SIP2Net | `sip2net_150` | ~29M |
| DC-Net | `dcnet_150` | ~37M |

## Loss Configurations

| Config name | Arguments |
|---|---|
| L2 Only | `--loss_l2_weight 1e7` |
| DA-cGAN Stage 2 | `--loss_ms_ssim_weight 8.4e6 --loss_tv_weight 1e3 --loss_focal_weight 1e7` |
| Hybrid L2+MSGSIM | `--loss_l2_weight 6e6 --loss_ms_gsim_weight 4e6` |
| SIP2Net | `--loss_l1_weight 1e7 --loss_ssim_weight 5e6 --loss_gdl_weight 1e4` |

## Citation

```bibtex
@article{yapar2026deployment1tx,
  title={Learning Coverage- and Power-Optimal Transmitter Placement from Building Maps: A Comparative Study of Direct and Indirect Neural Approaches},
  author={Yapar, {\c{C}}a{\u{g}}kan},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
