import os

import numpy as np

RAW_TO_TRAIN = {0: 0, 1: 1, 2: 2, 4: 3}
TRAIN_TO_RAW = {0: 0, 1: 1, 2: 2, 3: 4}
NUM_CLASSES = 4
MODALITIES = ("flair", "t1", "t1ce", "t2")

def _find_seg(pdir, name):
    std = os.path.join(pdir, f"{name}_seg.nii")
    if os.path.exists(std):
        return std
    for f in sorted(os.listdir(pdir)):
        if "seg" in f.lower() and f.lower().endswith(".nii"):
            return os.path.join(pdir, f)
    return None

def list_patients(root_dir):
    patients = []
    for name in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, name)
        if not os.path.isdir(pdir):
            continue
        has_modalities = all(
            os.path.exists(os.path.join(pdir, f"{name}_{m}.nii")) for m in MODALITIES
        )
        if has_modalities and _find_seg(pdir, name) is not None:
            patients.append(name)
    return patients

def split_patients(patients, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    order = np.array(patients)
    perm = rng.permutation(len(order))
    order = order[perm]
    n_val = max(1, int(round(len(order) * val_frac)))
    val_ids = list(order[:n_val])
    train_ids = list(order[n_val:])
    return train_ids, val_ids

def remap_labels(seg):
    out = np.zeros_like(seg, dtype=np.uint8)
    for raw, train in RAW_TO_TRAIN.items():
        out[seg == raw] = train
    return out

def unmap_labels(pred):
    out = np.zeros_like(pred, dtype=np.uint8)
    for train, raw in TRAIN_TO_RAW.items():
        out[pred == train] = raw
    return out

def load_patient(root_dir, patient):
    import nibabel as nib

    pdir = os.path.join(root_dir, patient)
    channels = []
    for m in MODALITIES:
        vol = nib.load(os.path.join(pdir, f"{patient}_{m}.nii")).get_fdata()
        channels.append(vol.astype(np.float32))
    image = np.stack(channels, axis=0)
    seg = nib.load(_find_seg(pdir, patient)).get_fdata()
    seg = np.round(seg).astype(np.uint8)
    return image, seg

def _foreground_bbox(seg):
    coords = np.argwhere(seg > 0) if (seg > 0).any() else None
    return coords

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    _TorchDataset = object

class BraTSDataset(_TorchDataset):

    def __init__(
        self,
        root_dir,
        patient_ids,
        norm_strategy="hybrid_percentile_zscore",
        patch_size=(96, 96, 96),
        training=True,
        augmentor=None,
        fg_prob=0.7,
        seed=0,
    ):
        self.root_dir = root_dir
        self.patient_ids = list(patient_ids)
        self.norm_strategy = norm_strategy
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        self.training = training
        self.augmentor = augmentor
        self.fg_prob = fg_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.patient_ids)

    def _crop(self, image, seg):
        from_shape = image.shape[1:]
        ph, pw, pd = self.patch_size
        H, W, D = from_shape

        pad = [
            (0, 0),
            (0, max(0, ph - H)),
            (0, max(0, pw - W)),
            (0, max(0, pd - D)),
        ]
        if any(p[1] > 0 for p in pad):
            image = np.pad(image, pad, mode="constant")
            seg = np.pad(seg, [(p[0], p[1]) for p in pad[1:]], mode="constant")
            H, W, D = image.shape[1:]

        if self.training and self.rng.random() < self.fg_prob and (seg > 0).any():
            center = np.argwhere(seg > 0)
            cy, cx, cz = center[self.rng.integers(0, len(center))]
        else:
            cy = self.rng.integers(ph // 2, H - ph // 2 + 1)
            cx = self.rng.integers(pw // 2, W - pw // 2 + 1)
            cz = self.rng.integers(pd // 2, D - pd // 2 + 1)

        y1 = int(np.clip(cy - ph // 2, 0, H - ph))
        x1 = int(np.clip(cx - pw // 2, 0, W - pw))
        z1 = int(np.clip(cz - pd // 2, 0, D - pd))
        img_c = image[:, y1:y1 + ph, x1:x1 + pw, z1:z1 + pd]
        seg_c = seg[y1:y1 + ph, x1:x1 + pw, z1:z1 + pd]
        return img_c, seg_c

    def __getitem__(self, idx):
        import torch

        from .normalization import normalize_volume

        patient = self.patient_ids[idx]
        image, seg_raw = load_patient(self.root_dir, patient)
        image = normalize_volume(image, self.norm_strategy)
        seg = remap_labels(seg_raw)

        if self.patch_size is not None:
            image, seg = self._crop(image, seg)

        if self.training and self.augmentor is not None:
            image, seg = self.augmentor(image, seg)

        image = np.ascontiguousarray(image, dtype=np.float32)
        seg = np.ascontiguousarray(seg, dtype=np.int64)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(seg),
            "patient": patient,
        }
