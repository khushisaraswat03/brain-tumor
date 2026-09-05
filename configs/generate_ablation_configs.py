import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(HERE, os.pardir, "config.yaml")
OUT_DIR = os.path.join(HERE, "generated")

NORM_STRATEGIES = [
    "zscore_brain",
    "percentile_clip",
    "white_stripe",
    "hybrid_percentile_zscore",
]

AUG_ARMS = [
    ("aug_no_aug",
     dict(enabled=False, spatial=False, intensity=False, batch_mixer="none")),
    ("aug_spatial_only",
     dict(enabled=True, spatial=True, intensity=False, batch_mixer="none")),
    ("aug_intensity_only",
     dict(enabled=True, spatial=False, intensity=True, batch_mixer="none")),
    ("aug_spatial_intensity",
     dict(enabled=True, spatial=True, intensity=True, batch_mixer="none")),
    ("aug_full_with_tumor_cutmix",
     dict(enabled=True, spatial=True, intensity=True,
          batch_mixer="tumor_aware_cutmix")),
]
REDUCED_AUG_NAMES = {"aug_no_aug", "aug_spatial_intensity",
                     "aug_full_with_tumor_cutmix"}

def load_base():
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)

def write_cfg(cfg, name):
    cfg = copy.deepcopy(cfg)
    cfg["experiment_name"] = name
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", action="store_true",
                    help="emit only the 3-arm reduced augmentation set")
    ap.add_argument("--best-norm", default="hybrid_percentile_zscore",
                    help="normalization held fixed for the augmentation axis")
    ap.add_argument("--fixed-aug-for-norm", default="aug_spatial_intensity",
                    help="augmentation arm held fixed for the normalization axis")
    args = ap.parse_args()

    base = load_base()
    fixed_aug = dict(AUG_ARMS)[args.fixed_aug_for_norm]
    written = []

    for norm in NORM_STRATEGIES:
        cfg = copy.deepcopy(base)
        cfg["data"]["norm_strategy"] = norm
        cfg["augmentation"] = copy.deepcopy(fixed_aug)
        written.append(write_cfg(cfg, f"norm_{norm}"))

    for name, aug in AUG_ARMS:
        if args.reduced and name not in REDUCED_AUG_NAMES:
            continue
        cfg = copy.deepcopy(base)
        cfg["data"]["norm_strategy"] = args.best_norm
        cfg["augmentation"] = copy.deepcopy(aug)
        written.append(write_cfg(cfg, name))

    print(f"Wrote {len(written)} configs to {OUT_DIR}/:")
    for p in written:
        print("  ", os.path.relpath(p))
    if args.reduced:
        print("\n(reduced augmentation axis: 3 arms)")

if __name__ == "__main__":
    main()
