"""Five-class LandCover.ai metrics; rows are truth, columns are predictions."""
import numpy as np

CLASSES = ('Background', 'Building', 'Woodland', 'Water', 'Road')


def confusion_counts(truth, prediction):
    """Accumulate integer pixels without merging background or rare classes."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError('Expected matching, nonempty masks')
    for value in (truth, prediction):
        if value.dtype.kind not in 'iu' or value.min() < 0 or value.max() >= 5:
            raise ValueError('Masks must contain integer class IDs 0..4')
    return np.bincount((5 * truth.astype(np.int64) + prediction).ravel(),
                       minlength=25).reshape(5, 5)


def metrics_from_counts(counts):
    """Pooled pixel IoU, macro over all five classes, and pixel accuracy.

    A class with zero union has IoU zero, with its absence reported separately.
    The actual validation cohort must contain every ground-truth class.
    Foreground-only mIoU is a secondary diagnostic, never the primary metric.
    """
    counts = np.asarray(counts)
    if (counts.shape != (5, 5) or counts.dtype.kind not in 'iu' or
            np.any(counts < 0) or int(counts.sum()) <= 0):
        raise ValueError('Expected nonnegative, nonempty integer 5x5 counts')
    true, predicted = counts.sum(1), counts.sum(0)
    diagonal = counts.diagonal()
    union = true + predicted - diagonal
    iou = np.divide(diagonal, union, out=np.zeros(5, dtype=float), where=union > 0)
    return dict(confusion=counts.tolist(), pixels=int(counts.sum()),
                truth_pixels=true.tolist(), prediction_pixels=predicted.tolist(),
                class_iou=dict(zip(CLASSES, iou.tolist())),
                absent_union=[CLASSES[i] for i in range(5) if union[i] == 0],
                miou=float(iou.mean()), foreground_miou=float(iou[1:].mean()),
                accuracy=float(diagonal.sum() / counts.sum()))


def segmentation_loss(logits, masks):
    """Unweighted 0.5 CE + 0.5 macro soft Dice over all five classes.

    This is an explicit local training assumption: HQ-UNet does not specify its
    loss. No binary flood loss/evaluator is called in this experiment.
    """
    import torch
    from torch.nn import functional as F
    if (logits.ndim != 4 or logits.shape[1] != 5 or masks.dtype != torch.long or
            tuple(masks.shape) != (logits.shape[0], *logits.shape[2:])):
        raise ValueError('Expected [B,5,H,W] logits and [B,H,W] long labels')
    if masks.numel() == 0 or int(masks.min()) < 0 or int(masks.max()) > 4:
        raise ValueError('Labels must be 0..4')
    probability = logits.softmax(1)
    target = F.one_hot(masks, num_classes=5).permute(0, 3, 1, 2).to(logits.dtype)
    axes = (0, 2, 3)
    dice = (2 * (probability * target).sum(axes) + 1e-6) / (
        probability.sum(axes) + target.sum(axes) + 1e-6)
    # CUDA's spatial NLL reduction has no deterministic implementation in the
    # inspected PyTorch runtime. Explicit log-probability selection implements
    # the same unweighted CE while retaining strict deterministic execution.
    cross_entropy = -F.log_softmax(logits, dim=1).gather(1, masks.unsqueeze(1)).mean()
    return .5 * cross_entropy + .5 * (1 - dice.mean())
