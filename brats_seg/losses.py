import torch
import torch.nn as nn
import torch.nn.functional as F

def _to_onehot(target, num_classes):
    if target.dim() == 5:
        return target.float()
    onehot = F.one_hot(target.long(), num_classes)
    return onehot.permute(0, 4, 1, 2, 3).float().contiguous()

class DiceLoss(nn.Module):

    def __init__(self, num_classes=4, include_background=False, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target = _to_onehot(target, self.num_classes).to(probs.dtype)

        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * target, dims)
        cardinality = torch.sum(probs + target, dims)
        dice = (2.0 * inter + self.smooth) / (cardinality + self.smooth)

        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()

class DiceCELoss(nn.Module):

    def __init__(self, num_classes=4, dice_weight=1.0, ce_weight=1.0,
                 include_background=False, class_weights=None):
        super().__init__()
        self.dice = DiceLoss(num_classes, include_background=include_background)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None,
        )

    def _ce(self, logits, target):
        if target.dim() == 5:
            logp = F.log_softmax(logits, dim=1)
            ce = -(target * logp)
            if self.class_weights is not None:
                ce = ce * self.class_weights.view(1, -1, 1, 1, 1)
            return ce.sum(dim=1).mean()
        return F.cross_entropy(logits, target.long(), weight=self.class_weights)

    def forward(self, logits, target):
        return (
            self.dice_weight * self.dice(logits, target)
            + self.ce_weight * self._ce(logits, target)
        )

def build_loss(cfg):
    l = cfg.get("loss", {})
    return DiceCELoss(
        num_classes=cfg.model.get("num_classes", 4),
        dice_weight=l.get("dice_weight", 1.0),
        ce_weight=l.get("ce_weight", 1.0),
        include_background=l.get("include_background", False),
        class_weights=l.get("class_weights", None),
    )
