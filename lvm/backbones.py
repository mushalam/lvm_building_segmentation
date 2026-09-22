#!/usr/bin/env python3
"""Alternative backbones for the Mask R-CNN detector.

Motivation. Runs 1-4 attributed the gap to SAM 3 to pretraining and scale
rather than mask-head design, and plan.md ranks "a stronger pretrained
backbone" as the only change likely to close 0.044. DINOv3 is the strongest
available self-supervised pretraining that still exposes a convolutional,
multi-stride feature hierarchy, so it drops into an FPN without the quadratic
attention cost a ViT would incur at 2048 px.

Parameter matching is deliberate. ConvNeXt-Tiny is 27.8M parameters against
ResNet-50's 25.6M, and both emit four stages at strides 4/8/16/32. Holding
capacity roughly fixed makes this a test of *pretraining quality* rather than
of model size -- the distinction runs 3 and 4 failed to preserve when they
varied learning rate and epoch count together.

One confound cannot be removed cheaply, and is stated rather than hidden:
torchvision's maskrcnn_resnet50_fpn_v2 ships COCO-pretrained FPN, RPN and
heads, whereas this path initialises all of those randomly. The DINOv3 arm is
therefore handicapped on head initialisation while advantaged on backbone
pretraining. The asymmetry is acceptable because it is directional -- if the
DINOv3 arm *wins*, the conclusion is safe a fortiori; if it loses, the result
is ambiguous and must be reported as such.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork, LastLevelMaxPool)

TIMM_BACKBONES = {
    "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "dinov3_convnext_small": "convnext_small.dinov3_lvd1689m",
    "dinov3_convnext_base": "convnext_base.dinov3_lvd1689m",
}


class TimmFPNBackbone(nn.Module):
    """timm features_only trunk + torchvision FPN.

    torchvision's detector expects an OrderedDict of feature maps and an
    `out_channels` attribute; timm returns a plain list. LastLevelMaxPool adds
    the fifth ("pool") level, so the number of maps matches the five anchor
    tuples the RPN is configured with -- a mismatch here is silent and costs
    recall at one scale.
    """

    def __init__(self, timm_name, out_channels=256, trainable_stages=4):
        super().__init__()
        import timm
        self.body = timm.create_model(timm_name, pretrained=True,
                                      features_only=True)
        chans = self.body.feature_info.channels()
        reductions = self.body.feature_info.reduction()
        assert len(chans) == 4, f"expected 4 stages, got {len(chans)}: {chans}"
        assert reductions == [4, 8, 16, 32], f"unexpected strides {reductions}"

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(chans), out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(), norm_layer=nn.BatchNorm2d)
        self.out_channels = out_channels

        if trainable_stages < 4:
            stages = list(self.body.children())
            for m in stages[:len(stages) - trainable_stages]:
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, x):
        feats = self.body(x)
        return self.fpn(OrderedDict((str(i), f) for i, f in enumerate(feats)))


def build_timm_maskrcnn(backbone, anchor_sizes, detections_per_img=400,
                        image_size=2048, num_classes=2):
    """Mask R-CNN on a timm backbone, with torchvision's v2 head configuration.

    Head widths, depths and norm layers mirror maskrcnn_resnet50_fpn_v2 so the
    only intended difference from the baseline is the trunk.
    """
    assert backbone in TIMM_BACKBONES, f"unknown backbone {backbone}"
    body = TimmFPNBackbone(TIMM_BACKBONES[backbone])

    from torchvision.models.detection.rpn import AnchorGenerator
    anchor_gen = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes))
    assert len(anchor_sizes) == 5, (
        f"FPN emits 5 maps, got {len(anchor_sizes)} anchor tuples")

    rpn_head = RPNHead(body.out_channels,
                       anchor_gen.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (body.out_channels, 7, 7), [256, 256, 256, 256], [1024],
        norm_layer=nn.BatchNorm2d)
    mask_head = MaskRCNNHeads(body.out_channels, [256, 256, 256, 256], 1,
                              norm_layer=nn.BatchNorm2d)

    return MaskRCNN(
        body, num_classes=num_classes,
        rpn_anchor_generator=anchor_gen, rpn_head=rpn_head,
        box_head=box_head, mask_head=mask_head,
        box_detections_per_img=detections_per_img,
        rpn_post_nms_top_n_train=3000, rpn_post_nms_top_n_test=3000,
        rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=4000,
        box_batch_size_per_image=512, rpn_batch_size_per_image=256,
        min_size=image_size, max_size=image_size)
