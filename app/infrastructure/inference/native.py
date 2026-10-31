"""Native ultralytics model loading and prediction."""

import logging

log = logging.getLogger(__name__)


def load_model(model_path, require_refinement=False):
    import os

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请检查 .env 中的模型路径，"
            f"或确认该权重文件已放置到正确路径。"
        )

    import torch
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("using device: %s", device)
    model = YOLO(model_path).to(device)
    from .our import adapt_our_model

    adapted = adapt_our_model(model)
    if require_refinement and not adapted:
        raise ValueError(
            "剑叶 OUR 权重必须包含 mask_refine。请将 .env 的 LEAF_MODEL_PATH "
            "改为 ./models_yolo/leaf/leaf-our-best.pt，不能继续使用旧版 exp59 权重。"
        )
    if adapted:
        log.info("OUR mask_refine enabled with official Ultralytics")
    return model


def predict(model, image_path, imgsz=640, conf=0.25, iou=0.7, retina_masks=False):
    kwargs = {
        "show": False,
        "save": False,
        "visualize": False,
        "conf": conf,
        "iou": iou,
        "retina_masks": retina_masks,
    }
    if imgsz:
        kwargs["imgsz"] = imgsz
    if getattr(model, "_rice_our", False):
        from .our import OurSegmentationPredictor

        kwargs["predictor"] = OurSegmentationPredictor
    return model(image_path, **kwargs)
