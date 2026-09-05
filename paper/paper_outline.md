# Paper Outline — Normalization & Augmentation Ablation for BraTS2020 3D U-Net

Working title (pick one):
- *"A Controlled Ablation of Intensity Normalization and Augmentation for 3D
  U-Net Brain Tumor Segmentation on BraTS2020"*
- *"Tumor-Aware CutMix: Class-Imbalance-Targeted Augmentation for Brain Tumor
  Segmentation"*

> **Integrity rule:** every number in Results/Tables must come from an actual
> `results/results_*.json` file. Never hand-write a plausible-looking value.
> Every citation must be independently verified (title/authors/venue/year) before
> submission — the starter list below is from memory, not verified.

## Abstract
Problem, method (3D U-Net + two-axis ablation + tumor-aware CutMix), headline
result (fill from results), one-line significance.

## 1. Introduction
- Clinical motivation: gliomas, why automated segmentation matters.
- Gap: normalization/augmentation choices are often adopted by default, rarely
  ablated in a controlled way on the same architecture.
- Contributions (3, bulleted): normalization ablation, augmentation ablation,
  tumor-aware CutMix.

## 2. Related Work
- U-Net and 3D U-Net; nnU-Net as the strong self-configuring baseline.
- BraTS challenge and its standard metrics (Dice, HD95).
- MRI intensity normalization / harmonization (z-score, percentile, WhiteStripe).
- Mixing augmentations: MixUp, CutMix; class-imbalance-aware augmentation.
- **[verify every citation before use]**

## 3. Dataset
- BraTS2020: 369 training subjects (293 HGG / 76 LGG), 4 modalities
  (T1/T1ce/T2/FLAIR), volumes 240×240×155.
- Labels {NCR/NET=1, ED=2, ET=4}; composite regions WT / TC / ET.
- **Split: patient-level** (state the exact val_frac and seed). No official
  test-set ground truth is public — evaluation is on a held-out patient split.
- Note subject 355's mislabeled mask handling.

## 4. Methodology
### 4.1 3D U-Net architecture
Encoder/decoder, instance norm, skip connections; patch-based training +
sliding-window inference (explain why).
### 4.2 Normalization strategies
Define all four precisely (brain-masked stats; the hybrid clip-then-zscore).
### 4.3 Augmentation strategies
Spatial (flip/rot90/elastic), intensity (gamma/bias-field/noise/blur), and the
batch mixers. **Tumor-aware CutMix**: give the centroid-guided algorithm and the
imbalance argument (a random box misses the <5% tumor voxels; centroid paste
guarantees a real tumor sub-region every time). Include a schematic figure.
### 4.4 Loss
Dice + cross-entropy; why CE alone fails under imbalance.

## 5. Experimental Setup
- Config table (patch size, base channels, depth, batch/accum, LR, schedule,
  epochs, early stopping) — and note where 4GB vs Kaggle settings differed.
- Metrics: per-region Dice, sensitivity, specificity, HD95.
- Seeds: state exactly how many per arm (be honest about compute limits).
- Hardware: local GTX 1650 for debugging; Kaggle T4/P100 for full runs.

## 6. Results
- **Table 1** — Normalization axis (4 arms): WT/TC/ET Dice + HD95, mean ± std.
- **Table 2** — Augmentation axis (best norm fixed): the arms, same columns.
- **Figure** — qualitative overlays (input / GT / prediction) for a GOOD case
  and a FAILURE case.
- State whether tumor-aware CutMix actually beat plain CutMix / no-mix — report
  the truth either way.

## 7. Discussion
Which normalization won and a plausible why; whether the augmentation gains are
meaningful vs the seed variance; where the model fails (small ET regions, etc.).

## 8. Limitations
Single architecture; non-official train/test split; patch-based not full-volume;
single dataset (BraTS2020); limited seeds on some arms; one subject excluded/
recovered.

## 9. Conclusion & Future Work
Recap findings; future: multi-dataset, full nnU-Net comparison, more seeds,
significance testing.

## References
Starter pointers (**verify each independently**): U-Net (Ronneberger 2015),
3D U-Net (Çiçek 2016), nnU-Net (Isensee 2021), BraTS (Menze 2015; Bakas 2017/2018),
WhiteStripe (Shinohara 2014), MixUp (Zhang 2018), CutMix (Yun 2019).

---
### Suggested venues (easiest → hardest)
arXiv preprint → student/regional IEEE conferences → BraTS workshop track (if a
window is open) → journals (Biomedical Signal Processing and Control, Journal of
Digital Imaging, Computers in Biology and Medicine, IEEE Access).

### Statistical rigor
Aim for ≥3 seeds on at least the best configuration; report mean ± std, and a
paired test (Wilcoxon / paired t-test) if seeds allow. State honestly what was
actually run.
