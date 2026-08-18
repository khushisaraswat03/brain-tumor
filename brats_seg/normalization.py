import numpy as np

_EPS = 1e-8

def _brain_mask(volume):
    return volume > 0

def zscore_brain(volume):
    volume = volume.astype(np.float32)
    mask = _brain_mask(volume)
    if mask.sum() == 0:
        return volume
    brain = volume[mask]
    mean = brain.mean()
    std = brain.std()
    if std < _EPS:
        return volume - mean
    return (volume - mean) / std

def percentile_clip(volume, low=0.5, high=99.5):
    volume = volume.astype(np.float32)
    mask = _brain_mask(volume)
    if mask.sum() == 0:
        return volume
    brain = volume[mask]
    lo = np.percentile(brain, low)
    hi = np.percentile(brain, high)
    if hi - lo < _EPS:
        return np.clip(volume, 0, 1)
    clipped = np.clip(volume, lo, hi)
    return (clipped - lo) / (hi - lo)

def white_stripe(volume, width=0.05):
    volume = volume.astype(np.float32)
    mask = _brain_mask(volume)
    if mask.sum() == 0:
        return volume
    brain = volume[mask]

    hist, edges = np.histogram(brain, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = centers[int(np.argmax(hist))]

    span = width * (brain.max() - brain.min() + _EPS)
    near = brain[np.abs(brain - mode) <= span]
    spread = near.std() if near.size > 1 else brain.std()
    if spread < _EPS:
        spread = brain.std() + _EPS
    return (volume - mode) / spread

def hybrid_percentile_zscore(volume, low=0.5, high=99.5):
    volume = volume.astype(np.float32)
    mask = _brain_mask(volume)
    if mask.sum() == 0:
        return volume
    brain = volume[mask]
    lo = np.percentile(brain, low)
    hi = np.percentile(brain, high)
    clipped = np.clip(volume, lo, hi)

    brain_clipped = clipped[mask]
    mean = brain_clipped.mean()
    std = brain_clipped.std()
    if std < _EPS:
        return clipped - mean
    return (clipped - mean) / std

_STRATEGIES = {
    "zscore_brain": zscore_brain,
    "percentile_clip": percentile_clip,
    "white_stripe": white_stripe,
    "hybrid_percentile_zscore": hybrid_percentile_zscore,
}

def normalize_volume(stack, strategy):
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown normalization strategy '{strategy}'. "
            f"Choose from {sorted(_STRATEGIES)}."
        )
    fn = _STRATEGIES[strategy]
    out = np.empty_like(stack, dtype=np.float32)
    for c in range(stack.shape[0]):
        out[c] = fn(stack[c])
    return out

def available_strategies():
    return sorted(_STRATEGIES)
