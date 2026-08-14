# Brain Tumor Segmentation — BraTS2020 (3D U-Net)

Final-year project: multimodal brain tumor segmentation on **BraTS2020** with a
**3D U-Net**, framed as a controlled ablation study with two axes and an original
augmentation contribution.

**Novel contributions**

1. **Normalization ablation** — 4 intensity-normalization strategies compared
   head-to-head (`zscore_brain`, `percentile_clip`, `white_stripe`,
   `hybrid_percentile_zscore`).
2. **Augmentation ablation** — standard spatial/intensity augmentations plus an
   original **tumor-aware CutMix** that centers the paste box on the donor's
   tumor centroid, directly targeting BraTS's extreme background/tumor class
   imbalance.

## Project layout

```
brats_seg/            # PyTorch package
  normalization.py    # ablation axis 1: 4 strategies + dispatcher
  augmentation.py     # ablation axis 2: per-sample augs + mixup/cutmix/tumor-aware CutMix
  dataset.py          # NIfTI loader, PATIENT-LEVEL split, patch cropping
  model.py            # configurable 3D U-Net (instance norm, skip connections)
  losses.py           # DiceLoss + DiceCELoss (hard & soft labels)
  metrics.py          # WT/TC/ET Dice, sensitivity, specificity, HD95
  train.py            # AMP + grad accumulation + cosine LR + early stopping
  evaluate.py         # sliding-window inference + JSON metric export
  utils.py            # config/seed/checkpoint helpers
config.yaml           # one run's settings (4GB-tuned defaults)
configs/generate_ablation_configs.py   # emits one config per ablation arm
results/              # results_*.json — the paper's evidence trail (committed)
paper/paper_outline.md
```

## Setup

```bash
pip install -r requirements.txt
```

**GPU / driver note.** Training needs CUDA. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If you have an old NVIDIA driver (e.g. 457.34 / CUDA 11.1), either update the
driver (recommended) or install the pinned fallback build:

```bash
pip install torch==1.10.1+cu111 --index-url https://download.pytorch.org/whl/cu111
```

If `cuda.is_available()` is `False`, training silently runs on CPU (50-100x
slower) — not practical. Do local **debugging** on CPU/small subset; run real
training on Kaggle.

## Recommended workflow (laptop + Kaggle + GitHub)

| Tool       | Role                                                              |
| ---------- | ----------------------------------------------------------------- |
| **Laptop** | Write/debug. Smoke-test on 2-4 patients (`data.max_patients`).    |
| **Kaggle** | Real training + the full ablation. Free T4/P100, ~30 GPU-hr/week. |
| **GitHub** | Single source of truth. Push from laptop, `git clone` on Kaggle.  |

On Kaggle: attach _"BraTS2020 Dataset (Training + Validation)"_ (by awsaf49),
set Accelerator = GPU, Internet = ON, then in the first cell `git clone` this
repo, `pip install -r requirements.txt`, and point `data.root_dir` at the mounted
dataset path (verify with `ls`).

## Running

Single experiment:

```bash
python -m brats_seg.train    --config config.yaml
python -m brats_seg.evaluate --config config.yaml \
    --checkpoint checkpoints/baseline/best.pt --split val
```

Fast local smoke test (few patients, CPU-OK): set `data.max_patients: 4` and
`train.epochs: 2` in a copy of `config.yaml`.

Full ablation:

```bash
python configs/generate_ablation_configs.py            # all 9 arms
# or the reduced 3-augmentation-arm set (~1/3 the compute):
python configs/generate_ablation_configs.py --reduced --best-norm hybrid_percentile_zscore

for cfg in configs/generated/*.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m brats_seg.train    --config "$cfg"
  python -m brats_seg.evaluate --config "$cfg" \
      --checkpoint "checkpoints/$name/best.pt" --split val
done
```

Commit the resulting `results/results_*.json` back to the repo — they are the
evidence behind every number in the paper.

## Configuration for a 4GB GPU (GTX 1650)

| Setting                   | Default (4GB) | Bigger GPU      |
| ------------------------- | ------------- | --------------- |
| `patch_size`              | `[96,96,96]`  | `[128,128,128]` |
| `base_channels`           | `8`           | `16-32`         |
| `depth`                   | `3`           | `4`             |
| `batch_size`              | `1`           | `2-4`           |
| `accumulate_grad_batches` | `8`           | `4`             |

If you still hit CUDA OOM, drop `patch_size` to `[64,64,64]`. Record which
settings each experiment used — it affects the ablation comparison.

## Key correctness choices

- **Patient-level split** (`dataset.split_patients`) — never split patches/slices
  of the same patient across train/val (that leaks and inflates scores).
- **HD95 + Dice** — Dice alone can look fine while boundaries are wrong.
- **Instance norm** — batch stats are unreliable at batch size 1-2.
- Subject **355**'s mislabeled mask (`W39_1998.09.19_Segm.nii`) is auto-recovered.

## Data / license

The dataset is **not** committed (multi-GB, registration-restricted). Use the
Kaggle mirror; don't re-upload it. See `LICENSE` before publishing.
