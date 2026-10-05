"""Mask Scoring R-CNN on torchvision's Mask R-CNN (Huang et al., CVPR 2019).

A MaskIoU head predicts each detection's mask IoU with its ground truth; at
inference the detection score becomes  class score x predicted mask IoU.
Mask R-CNN otherwise ranks by classification confidence alone, which says
nothing about outline quality. On r10 an oracle version of this re-ranking --
true mask IoU in place of the prediction -- lifts test AP 0.2176 -> 0.3135
(tools/oracle_rescore.py), so the ranking, not the masks, is where much of
the AP is lost.

Head, as in the paper: input is the 14x14 RoIAlign feature concatenated with
the predicted mask max-pooled from 28x28 to 14x14; four 3x3 convs (the last
stride 2) and three FC layers; one output (single class). Target, as in the
paper's simple form: IoU between the binarised predicted mask (logit > 0) and
the ground-truth mask projected into the same RoI at 28x28, for positive RoIs
only. L2 loss, weight 1.

Rather than fork RoIHeads.forward, the head is attached by wrapping two
modules -- mask_head keeps its input (the RoIAlign features), mask_predictor
keeps its output (the mask logits) -- and the subclass computes the loss from
those after torchvision's own forward has run. RoIs reach both wrappers in the
same order as the proposals / detections, so no bookkeeping is duplicated.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.roi_heads import RoIHeads, project_masks_on_boxes


class MaskIoUHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        c = in_channels + 1
        self.convs = nn.Sequential(
            nn.Conv2d(c, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.fcs = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 7 * 7, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Linear(1024, num_classes))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.fcs[-1].weight, std=0.01)

    def forward(self, roi_feats, mask_logit):
        """roi_feats (N,256,14,14); mask_logit (N,28,28) for each RoI's own class."""
        m = F.max_pool2d((mask_logit > 0).float()[:, None], 2)        # (N,1,14,14)
        return self.fcs(self.convs(torch.cat([roi_feats, m], 1)))


class _Keep(nn.Module):
    """Wraps a module and keeps its input or output in a plain dict.

    A plain dict, not a reference to the parent module: nn.Module registers any
    Module-valued attribute as a child, which would make the module tree cyclic
    and send state_dict() into infinite recursion."""
    def __init__(self, inner, cache, key, keep="output"):
        super().__init__()
        self.inner, self._cache, self._key, self._keep = inner, cache, key, keep

    def forward(self, x):
        y = self.inner(x)
        self._cache[self._key] = x if self._keep == "input" else y
        return y


class MaskScoringRoIHeads(RoIHeads):
    """RoIHeads whose mask branch also trains a MaskIoU head, and whose
    detections are scored  cls_score x predicted mask IoU  at inference."""

    @classmethod
    def from_roi_heads(cls, rh: RoIHeads):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.__dict__.update({k: v for k, v in rh.__dict__.items() if k not in ("_modules",)})
        new._modules = dict(rh._modules)
        new._cache = {}
        new.maskiou_head = MaskIoUHead()
        new.mask_head = _Keep(rh.mask_head, new._cache, "roi_feats", keep="input")
        new.mask_predictor = _Keep(rh.mask_predictor, new._cache, "mask_logits", keep="output")
        new.maskiou_weight = 1.0
        return new

    def forward(self, features, proposals, image_shapes, targets=None):
        captured = {}
        orig_loss = tv_roi_heads.maskrcnn_loss

        def loss_and_capture(mask_logits, mask_proposals, gt_masks, gt_labels, idxs):
            captured.update(proposals=mask_proposals, gt_masks=gt_masks,
                            gt_labels=gt_labels, idxs=idxs)
            return orig_loss(mask_logits, mask_proposals, gt_masks, gt_labels, idxs)

        tv_roi_heads.maskrcnn_loss = loss_and_capture
        try:
            result, losses = super().forward(features, proposals, image_shapes, targets)
        finally:
            tv_roi_heads.maskrcnn_loss = orig_loss

        logits, feats = self._cache.pop("mask_logits"), self._cache.pop("roi_feats")
        if self.training:
            labels = torch.cat([l[i] for l, i in zip(captured["gt_labels"], captured["idxs"])])
            own = logits[torch.arange(len(labels), device=labels.device), labels]
            M = logits.shape[-1]
            tgt = torch.cat([project_masks_on_boxes(m, p, i, M) for m, p, i in
                             zip(captured["gt_masks"], captured["proposals"], captured["idxs"])])
            with torch.no_grad():
                pred_b, gt_b = own > 0, tgt > 0.5
                inter = (pred_b & gt_b).flatten(1).sum(1).float()
                union = (pred_b | gt_b).flatten(1).sum(1).float()
                iou_t = torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))
            if len(own):
                pred = self.maskiou_head(feats, own.detach())[:, 0]
                losses["loss_maskiou"] = self.maskiou_weight * F.mse_loss(pred, iou_t)
            else:
                losses["loss_maskiou"] = logits.sum() * 0
        else:
            n = [len(r["scores"]) for r in result]
            if sum(n):
                labels = torch.cat([r["labels"] for r in result])
                own = logits[torch.arange(len(labels), device=labels.device), labels]
                iou = self.maskiou_head(feats, own)[:, 0].clamp(0, 1)
                for r, piece in zip(result, iou.split(n)):
                    r["mask_iou"] = piece
                    r["scores_cls"] = r["scores"]
                    r["scores"] = r["scores"] * piece
                # re-rank: torchvision returns detections sorted by class score
                for r in result:
                    o = torch.argsort(r["scores"], descending=True)
                    for k in ("boxes", "labels", "scores", "masks", "mask_iou", "scores_cls"):
                        r[k] = r[k][o]
        return result, losses


def add_mask_scoring(model):
    """Replace a torchvision Mask R-CNN's roi_heads with MaskScoringRoIHeads."""
    model.roi_heads = MaskScoringRoIHeads.from_roi_heads(model.roi_heads)
    return model
