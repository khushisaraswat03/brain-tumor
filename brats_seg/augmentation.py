import numpy as np

class Augmentor:

    def __init__(
        self,
        rng=None,
        spatial=True,
        intensity=True,
        p_flip=0.5,
        p_rot90=0.5,
        p_elastic=0.3,
        p_gamma=0.3,
        p_bias=0.3,
        p_noise=0.3,
        p_blur=0.2,
        elastic_alpha=10.0,
        elastic_sigma=4.0,
        gamma_range=(0.7, 1.5),
        bias_strength=0.3,
        noise_std=0.05,
        blur_sigma=0.8,
    ):
        self.rng = rng or np.random.default_rng()
        self.spatial = spatial
        self.intensity = intensity
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_elastic = p_elastic
        self.p_gamma = p_gamma
        self.p_bias = p_bias
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.gamma_range = gamma_range
        self.bias_strength = bias_strength
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma

    def __call__(self, image, label):
        if self.spatial:
            image, label = self._spatial(image, label)
        if self.intensity:
            image = self._intensity(image)
        return image, label

    def _spatial(self, image, label):
        for ax in (1, 2, 3):
            if self.rng.random() < self.p_flip:
                image = np.flip(image, axis=ax)
                label = np.flip(label, axis=ax - 1)
        if self.rng.random() < self.p_rot90:
            planes = [(1, 2), (1, 3), (2, 3)]
            k = int(self.rng.integers(1, 4))
            ax = planes[int(self.rng.integers(0, 3))]
            image = np.rot90(image, k=k, axes=ax)
            label = np.rot90(label, k=k, axes=(ax[0] - 1, ax[1] - 1))
        if self.rng.random() < self.p_elastic:
            image, label = self._elastic(image, label)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def _elastic(self, image, label):
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
        except ImportError:
            return image, label

        shape = image.shape[1:]
        disp = [
            gaussian_filter(
                (self.rng.random(shape) * 2 - 1), self.elastic_sigma
            )
            * self.elastic_alpha
            for _ in range(3)
        ]
        coords = np.meshgrid(
            *[np.arange(s) for s in shape], indexing="ij"
        )
        indices = [np.clip(c + d, 0, s - 1) for c, d, s in zip(coords, disp, shape)]

        warped_img = np.stack(
            [
                map_coordinates(image[c], indices, order=1, mode="nearest")
                for c in range(image.shape[0])
            ]
        )
        warped_lbl = map_coordinates(
            label, indices, order=0, mode="nearest"
        )
        return warped_img.astype(image.dtype), warped_lbl.astype(label.dtype)

    def _intensity(self, image):
        image = image.copy()
        if self.rng.random() < self.p_gamma:
            image = self._gamma(image)
        if self.rng.random() < self.p_bias:
            image = self._bias_field(image)
        if self.rng.random() < self.p_noise:
            image = image + self.rng.normal(0, self.noise_std, image.shape)
        if self.rng.random() < self.p_blur:
            image = self._blur(image)
        return image.astype(np.float32)

    def _gamma(self, image):
        out = np.empty_like(image)
        g = self.rng.uniform(*self.gamma_range)
        for c in range(image.shape[0]):
            ch = image[c]
            lo, hi = ch.min(), ch.max()
            if hi - lo < 1e-8:
                out[c] = ch
                continue
            norm = (ch - lo) / (hi - lo)
            out[c] = np.power(norm, g) * (hi - lo) + lo
        return out

    def _bias_field(self, image):
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return image
        shape = image.shape[1:]
        field = self.rng.random(shape)
        field = gaussian_filter(field, sigma=max(shape) / 4.0)
        field = 1.0 + self.bias_strength * (2 * (field - field.min()) /
                                            (field.max() - field.min() + 1e-8) - 1)
        return image * field[None]

    def _blur(self, image):
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return image
        sigma = self.rng.uniform(0, self.blur_sigma)
        out = np.empty_like(image)
        for c in range(image.shape[0]):
            out[c] = gaussian_filter(image[c], sigma)
        return out

def _to_soft(label, num_classes):
    label = np.asarray(label)
    if label.ndim == 4 and label.shape[0] == num_classes:
        return label.astype(np.float32)
    onehot = np.zeros((num_classes,) + label.shape, dtype=np.float32)
    for k in range(num_classes):
        onehot[k] = (label == k)
    return onehot

def mixup_batch(images, labels, num_classes, rng=None, alpha=0.2):
    rng = rng or np.random.default_rng()
    n = images.shape[0]
    perm = rng.permutation(n)
    lam = float(rng.beta(alpha, alpha))

    soft = np.stack([_to_soft(labels[i], num_classes) for i in range(n)])
    mixed_img = lam * images + (1 - lam) * images[perm]
    mixed_lbl = lam * soft + (1 - lam) * soft[perm]
    return mixed_img.astype(np.float32), mixed_lbl.astype(np.float32)

def _rand_bbox(shape, lam, rng):
    H, W, D = shape
    cut = (1.0 - lam) ** (1.0 / 3.0)
    ch, cw, cd = int(H * cut), int(W * cut), int(D * cut)
    cy, cx, cz = (
        int(rng.integers(0, H)),
        int(rng.integers(0, W)),
        int(rng.integers(0, D)),
    )
    y1, y2 = np.clip([cy - ch // 2, cy + ch // 2], 0, H)
    x1, x2 = np.clip([cx - cw // 2, cx + cw // 2], 0, W)
    z1, z2 = np.clip([cz - cd // 2, cz + cd // 2], 0, D)
    return int(y1), int(y2), int(x1), int(x2), int(z1), int(z2)

def cutmix_batch(images, labels, num_classes, rng=None, alpha=1.0):
    rng = rng or np.random.default_rng()
    n = images.shape[0]
    perm = rng.permutation(n)
    lam = float(rng.beta(alpha, alpha))
    spatial = images.shape[2:]

    soft = np.stack([_to_soft(labels[i], num_classes) for i in range(n)])
    out_img = images.copy()
    out_lbl = soft.copy()
    y1, y2, x1, x2, z1, z2 = _rand_bbox(spatial, lam, rng)
    out_img[:, :, y1:y2, x1:x2, z1:z2] = images[perm][:, :, y1:y2, x1:x2, z1:z2]
    out_lbl[:, :, y1:y2, x1:x2, z1:z2] = soft[perm][:, :, y1:y2, x1:x2, z1:z2]
    return out_img.astype(np.float32), out_lbl.astype(np.float32)

def tumor_aware_cutmix_batch(
    images, labels, num_classes, rng=None, box_scale=0.4, background_label=0
):
    rng = rng or np.random.default_rng()
    n = images.shape[0]
    perm = rng.permutation(n)
    H, W, D = images.shape[2:]
    bh, bw, bd = int(H * box_scale), int(W * box_scale), int(D * box_scale)

    soft = np.stack([_to_soft(labels[i], num_classes) for i in range(n)])
    out_img = images.copy()
    out_lbl = soft.copy()

    for i in range(n):
        donor = perm[i]
        donor_lbl = labels[donor]
        if np.asarray(donor_lbl).ndim == 4:
            donor_hard = np.argmax(donor_lbl, axis=0)
        else:
            donor_hard = np.asarray(donor_lbl)

        tumor_vox = np.argwhere(donor_hard != background_label)
        if tumor_vox.shape[0] > 0:
            cy, cx, cz = tumor_vox.mean(axis=0).astype(int)
        else:
            cy = int(rng.integers(0, H))
            cx = int(rng.integers(0, W))
            cz = int(rng.integers(0, D))

        y1, y2 = np.clip([cy - bh // 2, cy + bh // 2], 0, H)
        x1, x2 = np.clip([cx - bw // 2, cx + bw // 2], 0, W)
        z1, z2 = np.clip([cz - bd // 2, cz + bd // 2], 0, D)

        out_img[i, :, y1:y2, x1:x2, z1:z2] = images[donor, :, y1:y2, x1:x2, z1:z2]
        out_lbl[i, :, y1:y2, x1:x2, z1:z2] = soft[donor, :, y1:y2, x1:x2, z1:z2]

    return out_img.astype(np.float32), out_lbl.astype(np.float32)

BATCH_MIXERS = {
    "mixup": mixup_batch,
    "cutmix": cutmix_batch,
    "tumor_aware_cutmix": tumor_aware_cutmix_batch,
}
