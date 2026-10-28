"""剑叶 OUR 检测头适配（移植自 RiceAnylaze_V2.0）。

背景
----
剑叶权重 ``leaf-our.pt`` 的检测头里带一个训练好的 ``mask_refine`` 子模块。
官方 ``Segment.forward`` 只做 ``p = self.proto(x[0])``，**不会调用** ``mask_refine``：

    class Segment(Detect):
        def forward(self, x):
            p = self.proto(x[0])          # ← mask_refine 被跳过
            ...

反序列化不会报错，但原型掩膜没有经过 refinement，logits 基本落在 0 附近，
``logits > 0`` 的二值化结果就是一片噪点/矩形——也就是页面上的“分割异常”。

这里只对**含 mask_refine 的这一个模型实例**把 head 的类换成 ``OurSegment``，
不改 ultralytics 安装目录、也不改全局 ``Segment``；茎秆模型（yolov8s.pt）走原逻辑。

来源：RiceAnylaze_V2.0/app/infrastructure/inference/our.py（commit 8858ac1）。
适配验证的版本为 ultralytics==8.3.27（见 requirements）。
"""

from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.converter import merge_multi_segment
from ultralytics.engine.results import Masks, Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.utils import ops


class OurSegment(Segment):
    """标准 Segment 头，额外执行训练好的原型掩膜 refinement。"""

    def forward(self, x):
        p = self.mask_refine(self.proto(x[0]))
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1)
                        for i in range(self.nl)], 2)
        detection = Detect.forward(self, x)
        if self.training:
            return detection, mc, p
        if self.export:
            return torch.cat([detection, mc], 1), p
        return torch.cat([detection[0], mc], 1), (detection[1], mc, p)


class OurMasks(Masks):
    """保留参考运行时的“多连通域合并”轮廓，小维管束不会被拆碎。"""

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        segments = []
        for mask in self.data.byte().cpu().numpy():
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(contours) > 1:
                points = np.concatenate(merge_multi_segment([c.reshape(-1, 2) for c in contours]))
            elif contours:
                points = contours[0].reshape(-1, 2)
            else:
                points = np.zeros((0, 2))
            segments.append(ops.scale_coords(self.data.shape[1:], points.astype("float32"),
                                             self.orig_shape, normalize=False))
        return segments


def scale_our_masks(masks, shape):
    """按参考运行时的方式，对称裁掉 letterbox 留白后再缩放。"""
    height, width = masks.shape[-2:]
    if (height, width) == tuple(shape):
        return masks
    gain = min(height / shape[0], width / shape[1])
    pad_w = (width - shape[1] * gain) / 2
    pad_h = (height - shape[0] * gain) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = height - round(pad_h + 0.1), width - round(pad_w + 0.1)
    return F.interpolate(masks[..., top:bottom, left:right].float(), shape,
                         mode="bilinear", align_corners=False)


def _crop_masks(masks, boxes):
    """小批量 CPU 走参考运行时的取整裁剪，其余交给 ops.crop_mask。"""
    if len(masks) < 50 and not masks.is_cuda:
        for index, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[index, :y1] = 0
            masks[index, y2:] = 0
            masks[index, :, :x1] = 0
            masks[index, :, x2:] = 0
        return masks
    return ops.crop_mask(masks, boxes)


def _process_masks(proto, coefficients, boxes, shape, retina):
    channels, height, width = proto.shape
    masks = (coefficients @ proto.float().view(channels, -1)).view(-1, height, width)
    if retina:
        masks = scale_our_masks(masks[None], shape)[0]
        masks = _crop_masks(masks, boxes)
    else:
        ratios = boxes.new_tensor([width / shape[1], height / shape[0]] * 2)
        masks = _crop_masks(masks, boxes * ratios)
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]
    return masks.gt_(0).byte()


class OurSegmentationPredictor(SegmentationPredictor):
    """只给 OUR 模型用的掩膜对齐 + 空掩膜过滤后处理。"""

    def postprocess(self, preds, img, orig_imgs):
        detections = ops.non_max_suppression(
            preds[0], self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            nc=len(self.model.names), classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        results = []
        for pred, orig_img, path, proto in zip(detections, orig_imgs, self.batch[0], protos):
            masks = None
            if len(pred):
                if self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = _process_masks(proto, pred[:, 6:], pred[:, :4],
                                           orig_img.shape[:2], retina=True)
                else:
                    masks = _process_masks(proto, pred[:, 6:], pred[:, :4],
                                           img.shape[2:], retina=False)
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                keep = masks.amax((-2, -1)) > 0
                if not all(keep):
                    pred, masks = pred[keep], masks[keep]
            result = Results(orig_img, path=path, names=self.model.names,
                             boxes=pred[:, :6])
            if masks is not None:
                result.masks = OurMasks(masks, orig_img.shape[:2])
            results.append(result)
        return results


def adapt_our_model(model):
    """给已加载的 OUR 模型启用 refinement；返回是否发生了适配。"""
    import ultralytics

    heads = [m for m in model.model.modules() if isinstance(m, Segment)
             and hasattr(m, "mask_refine")]
    if not heads:
        return False
    if ultralytics.__version__ != "8.3.27":
        raise RuntimeError("OUR 适配已验证的 Ultralytics 版本为 8.3.27，请按 requirements 安装依赖")
    for head in heads:
        if not isinstance(head.mask_refine, torch.nn.Module):
            raise ValueError("OUR 权重中的 mask_refine 结构无效")
        head.__class__ = OurSegment
    model._rice_our = True
    return True
