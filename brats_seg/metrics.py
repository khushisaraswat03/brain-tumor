import numpy as np

REGIONS = ("WT", "TC", "ET")
_REGION_LABELS = {
    "WT": (1, 2, 4),
    "TC": (1, 4),
    "ET": (4,),
}

def region_mask(label_map, region):
    labels = _REGION_LABELS[region]
    return np.isin(label_map, labels)

def dice_score(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)

def sensitivity(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    denom = tp + fn
    if denom == 0:
        return float("nan")
    return float(tp / denom)

def specificity(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    denom = tn + fp
    if denom == 0:
        return float("nan")
    return float(tn / denom)

def hd95(pred_mask, gt_mask, spacing=None):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("nan")

    if spacing is None:
        spacing = (1.0,) * pred_mask.ndim

    def surface(mask):
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask)
        return mask & ~eroded

    surf_pred = surface(pred_mask)
    surf_gt = surface(gt_mask)

    dt_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    dist_pred_to_gt = dt_gt[surf_pred]
    dist_gt_to_pred = dt_pred[surf_gt]
    all_dist = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    if all_dist.size == 0:
        return 0.0
    return float(np.percentile(all_dist, 95))

def evaluate_case(pred_label, gt_label, spacing=None):
    results = {}
    for region in REGIONS:
        p = region_mask(pred_label, region)
        g = region_mask(gt_label, region)
        results[region] = {
            "dice": dice_score(p, g),
            "sensitivity": sensitivity(p, g),
            "specificity": specificity(p, g),
            "hd95": hd95(p, g, spacing=spacing),
        }
    return results

def aggregate_cases(case_results):
    agg = {}
    metrics = ("dice", "sensitivity", "specificity", "hd95")
    for region in REGIONS:
        agg[region] = {}
        for m in metrics:
            vals = np.array([c[region][m] for c in case_results], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                agg[region][m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            else:
                agg[region][m] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "n": int(vals.size),
                }
    return agg
