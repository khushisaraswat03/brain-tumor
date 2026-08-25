import argparse
import json
import os

import numpy as np
import torch

from .dataset import (
    BraTSDataset, list_patients, split_patients, unmap_labels,
)
from .metrics import REGIONS, aggregate_cases, evaluate_case
from .model import build_model
from .utils import load_config, load_checkpoint

def _gaussian_weight(patch_size, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grid = np.meshgrid(*coords, indexing="ij")
    dist2 = sum(g ** 2 for g in grid)
    w = np.exp(-dist2 / (2 * sigma_scale ** 2))
    return w.astype(np.float32)

@torch.no_grad()
def sliding_window_inference(model, image, patch_size, num_classes, device,
                             overlap=0.5):
    model.eval()
    C, H, W, D = image.shape
    ph, pw, pd = patch_size
    pad = [(0, max(0, ph - H)), (0, max(0, pw - W)), (0, max(0, pd - D))]
    img = np.pad(image.numpy(), [(0, 0)] + pad, mode="constant")
    _, H2, W2, D2 = img.shape

    step = [max(1, int(p * (1 - overlap))) for p in patch_size]
    ys = list(range(0, max(1, H2 - ph + 1), step[0])) or [0]
    xs = list(range(0, max(1, W2 - pw + 1), step[1])) or [0]
    zs = list(range(0, max(1, D2 - pd + 1), step[2])) or [0]
    if ys[-1] != H2 - ph:
        ys.append(H2 - ph)
    if xs[-1] != W2 - pw:
        xs.append(W2 - pw)
    if zs[-1] != D2 - pd:
        zs.append(D2 - pd)

    logits_sum = np.zeros((num_classes, H2, W2, D2), np.float32)
    weight_sum = np.zeros((H2, W2, D2), np.float32)
    gw = _gaussian_weight(patch_size)

    for y in ys:
        for x in xs:
            for z in zs:
                patch = img[:, y:y+ph, x:x+pw, z:z+pd]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                out = torch.softmax(model(t), dim=1)[0].cpu().numpy()
                logits_sum[:, y:y+ph, x:x+pw, z:z+pd] += out * gw
                weight_sum[y:y+ph, x:x+pw, z:z+pd] += gw

    weight_sum[weight_sum == 0] = 1.0
    probs = logits_sum / weight_sum
    pred = np.argmax(probs, axis=0)[:H, :W, :D]
    return pred.astype(np.uint8)

def evaluate(cfg_path, checkpoint, split="val"):
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    root = cfg.data.root_dir
    patch = tuple(cfg.data.get("patch_size", [96, 96, 96]))
    num_classes = cfg.model.get("num_classes", 4)

    patients = list_patients(root)
    if cfg.data.get("max_patients"):
        patients = patients[: cfg.data.max_patients]
    train_ids, val_ids = split_patients(
        patients, val_frac=cfg.data.get("val_frac", 0.15), seed=seed
    )
    ids = val_ids if split == "val" else train_ids

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_dice {ckpt.get('val_dice')})")

    ds = BraTSDataset(root, ids, cfg.data.get("norm_strategy",
                      "hybrid_percentile_zscore"), patch_size=None,
                      training=False, seed=seed)

    case_results = []
    for i in range(len(ds)):
        sample = ds[i]
        pred = sliding_window_inference(
            model, sample["image"], patch, num_classes, device,
            overlap=cfg.get("eval", {}).get("overlap", 0.5),
        )
        gt = sample["label"].numpy()
        res = evaluate_case(unmap_labels(pred), unmap_labels(gt))
        case_results.append(res)
        wt = res["WT"]["dice"]
        print(f"  [{i+1}/{len(ds)}] {sample['patient']} WT_dice={wt:.4f}")

    agg = aggregate_cases(case_results)
    print("\n=== Aggregate ({} cases, split={}) ===".format(len(case_results), split))
    for r in REGIONS:
        d = agg[r]["dice"]
        h = agg[r]["hd95"]
        print(f"  {r}: Dice {d['mean']:.4f} ± {d['std']:.4f} | "
              f"HD95 {h['mean']:.2f} ± {h['std']:.2f}")

    out = {
        "experiment": cfg.get("experiment_name", "default"),
        "split": split,
        "num_cases": len(case_results),
        "norm_strategy": cfg.data.get("norm_strategy"),
        "augmentation": cfg.get("augmentation", {}),
        "seed": seed,
        "aggregate": agg,
        "per_case": case_results,
    }
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join(
        "results", f"results_{cfg.get('experiment_name','default')}_{split}.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    args = ap.parse_args()
    evaluate(args.config, args.checkpoint, args.split)

if __name__ == "__main__":
    main()
