import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import Augmentor, BATCH_MIXERS
from .dataset import (BraTSDataset, NUM_CLASSES, list_patients, split_patients,
                      worker_init_fn)
from .losses import build_loss
from .metrics import REGIONS, aggregate_cases, evaluate_case
from .model import build_model
from .dataset import unmap_labels
from .utils import load_checkpoint, load_config, save_checkpoint, set_seed, _to_plain

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

def train(cfg_path, resume=False, max_hours=None):
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
        patients, val_frac=cfg.data.get("val_frac", 0.15),
        seed=cfg.data.get("split_seed", 42)
    )
    import hashlib as _h
    _fp = _h.md5("|".join(sorted(val_ids)).encode()).hexdigest()[:8]
    print(f"  patients: {len(patients)} (train={len(train_ids)}, val={len(val_ids)}) split_seed={cfg.data.get('split_seed', 42)} fingerprint={_fp}")

    norm = cfg.data.get("norm_strategy", "hybrid_percentile_zscore")
    patch = tuple(cfg.data.get("patch_size", [96, 96, 96]))
    aug = build_augmentor(cfg, seed)

    train_ds = BraTSDataset(root, train_ids, norm, patch, training=True,
                            augmentor=aug, seed=seed)
    val_ds = BraTSDataset(root, val_ids, norm, patch, training=False, seed=seed)
    num_workers = cfg.train.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.get("batch_size", 1),
                              shuffle=True, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=False,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.get("batch_size", 1),
                            shuffle=False, num_workers=num_workers,
                            worker_init_fn=worker_init_fn)

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
    start_epoch = 0
    last_path = os.path.join(ckpt_dir, "last.pt")

    if resume and os.path.exists(last_path):
        ck = load_checkpoint(last_path, map_location="cpu")
        if "optimizer" not in ck:
            raise SystemExit(
                f"{last_path} predates resume support (no optimizer state). "
                "Delete it and restart, or finish the run in one session."
            )
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        mix_rng.bit_generator.state = ck["mix_rng"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        best_dice = ck["best_dice"]
        bad_epochs = ck["bad_epochs"]
        start_epoch = ck["epoch"]
        if start_epoch >= epochs:
            print(f"[{exp}] already complete ({start_epoch}/{epochs} epochs). Nothing to do.")
            return best_dice
        print(f"  resumed from {last_path}: {start_epoch}/{epochs} epochs done, "
              f"best val_dice={best_dice:.4f}")
    elif resume:
        print(f"  --resume given but {last_path} not found; starting from scratch")

    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        train_ds.epoch = epoch
        if num_workers == 0:
            train_ds.reseed(epoch)
            val_ds.reseed(0)
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

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            bad_epochs = 0
        else:
            bad_epochs += 1

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "val_dice": val_dice,
            "best_dice": best_dice,
            "bad_epochs": bad_epochs,
            "mix_rng": mix_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "config": _to_plain(cfg),
        }
        save_checkpoint(state, last_path)
        if is_best:
            save_checkpoint(
                {k: state[k] for k in ("model", "epoch", "val_dice", "config")},
                os.path.join(ckpt_dir, "best.pt"),
            )
            print(f"    -> new best val_dice={best_dice:.4f} (saved best.pt)")

        if bad_epochs >= patience:
            print(f"  early stopping at epoch {epoch+1} (no gain in {patience})")
            break

        done = epoch + 1 - start_epoch
        elapsed = time.time() - t0
        if max_hours and elapsed + elapsed / done > max_hours * 3600:
            print(f"  time budget {max_hours}h reached after epoch {epoch+1}/{epochs} "
                  f"({elapsed/3600:.2f}h this session) — rerun with --resume to continue")
            break

    print(f"[{exp}] done. best val_dice={best_dice:.4f}")
    return best_dice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt in the experiment's checkpoint dir")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this many hours elapse this session")
    args = ap.parse_args()
    train(args.config, resume=args.resume, max_hours=args.max_hours)

if __name__ == "__main__":
    main()
