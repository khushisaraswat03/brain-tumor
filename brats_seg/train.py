import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import Augmentor, BATCH_MIXERS
from .dataset import BraTSDataset, NUM_CLASSES, list_patients, split_patients
from .losses import build_loss
from .metrics import REGIONS, aggregate_cases, evaluate_case
from .model import build_model
from .dataset import unmap_labels
from .utils import load_config, save_checkpoint, set_seed, _to_plain

def build_augmentor(cfg, seed):
    a = cfg.get("augmentation", {})
    if not a.get("enabled", True):
        return None
    if not a.get("spatial", True) and not a.get("intensity", True):
        return None
    return Augmentor(
        rng=np.random.default_rng(seed),
        spatial=a.get("spatial", True),
        intensity=a.get("intensity", True),
    )

def apply_batch_mixer(images, labels, cfg, rng):
    name = cfg.get("augmentation", {}).get("batch_mixer", None)
    if not name or name == "none":
        return images, labels
    mixer = BATCH_MIXERS[name]
    img_np = images.numpy()
    lbl_np = labels.numpy()
    mi, ml = mixer(img_np, lbl_np, NUM_CLASSES, rng=rng)
    return torch.from_numpy(mi), torch.from_numpy(ml)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, n = 0.0, 0
    dice_accum = {r: [] for r in REGIONS}
    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        logits = model(img)
        total_loss += loss_fn(logits, lbl).item()
        n += 1
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        gt = lbl.cpu().numpy()
        for b in range(pred.shape[0]):
            res = evaluate_case(unmap_labels(pred[b]), unmap_labels(gt[b]))
            for r in REGIONS:
                dice_accum[r].append(res[r]["dice"])
    mean_dice = float(np.mean([np.mean(dice_accum[r]) for r in REGIONS]))
    return total_loss / max(1, n), mean_dice

def train(cfg_path):
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    exp = cfg.get("experiment_name", "default")
    ckpt_dir = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), exp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[{exp}] device={device} seed={seed}")

    root = cfg.data.root_dir
    patients = list_patients(root)
    if cfg.data.get("max_patients"):
        patients = patients[: cfg.data.max_patients]
    train_ids, val_ids = split_patients(
        patients, val_frac=cfg.data.get("val_frac", 0.15), seed=seed
    )
    print(f"  patients: {len(patients)} (train={len(train_ids)}, val={len(val_ids)})")

    norm = cfg.data.get("norm_strategy", "hybrid_percentile_zscore")
    patch = tuple(cfg.data.get("patch_size", [96, 96, 96]))
    aug = build_augmentor(cfg, seed)

    train_ds = BraTSDataset(root, train_ids, norm, patch, training=True,
                            augmentor=aug, seed=seed)
    val_ds = BraTSDataset(root, val_ids, norm, patch, training=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.get("batch_size", 1),
                              shuffle=True, num_workers=cfg.train.get("num_workers", 2),
                              pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.get("batch_size", 1),
                            shuffle=False, num_workers=cfg.train.get("num_workers", 2))

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    lr = cfg.train.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.train.get("weight_decay", 1e-5))
    epochs = cfg.train.get("epochs", 100)
    accum = cfg.train.get("accumulate_grad_batches", 1)
    use_amp = cfg.train.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mix_rng = np.random.default_rng(seed + 1)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch
    )

    best_dice, patience, bad_epochs = -1.0, cfg.train.get("early_stop_patience", 20), 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running = 0.0
        for it, batch in enumerate(train_loader):
            img, lbl = apply_batch_mixer(batch["image"], batch["label"], cfg, mix_rng)
            img = img.to(device)
            lbl = lbl.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img)
                loss = loss_fn(logits, lbl) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        print(f"  epoch {epoch+1:3d}/{epochs} "
              f"train_loss={running/max(1,len(train_loader)):.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        state = {"model": model.state_dict(), "epoch": epoch,
                 "val_dice": val_dice, "config": _to_plain(cfg)}
        save_checkpoint(state, os.path.join(ckpt_dir, "last.pt"))
        if val_dice > best_dice:
            best_dice = val_dice
            bad_epochs = 0
            save_checkpoint(state, os.path.join(ckpt_dir, "best.pt"))
            print(f"    -> new best val_dice={best_dice:.4f} (saved best.pt)")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch+1} (no gain in {patience})")
                break

    print(f"[{exp}] done. best val_dice={best_dice:.4f}")
    return best_dice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
