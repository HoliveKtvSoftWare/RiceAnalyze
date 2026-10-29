"""图片缩略图工具。

记录条只需要 176×68 的缩略图，如果直接加载 1MB 左右的标注图，
一次拉取十几张就会造成明显的卡顿和闪烁。这里统一生成并缓存
`*.thumb.jpg`，历史接口只返回缩略图 URL；查看大图仍使用原图。
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

THUMB_SUFFIX = ".thumb.jpg"
DEFAULT_MAX_SIZE = 360
DEFAULT_QUALITY = 78


def thumbnail_path_for(image_path: Optional[str]) -> Optional[str]:
    """根据原图/标注图路径推导缩略图路径。"""
    if not image_path:
        return None
    return f"{os.path.splitext(image_path)[0]}{THUMB_SUFFIX}"


def ensure_thumbnail(
    image_path: Optional[str],
    max_size: int = DEFAULT_MAX_SIZE,
    quality: int = DEFAULT_QUALITY,
) -> Optional[str]:
    """
    生成或复用缩略图，返回缩略图路径（失败时返回 None）。

    缩略图存在且比源文件新时直接复用，避免每次请求都重新解码图片。
    """
    if not image_path or not os.path.exists(image_path):
        return None

    thumb_path = thumbnail_path_for(image_path)
    if not thumb_path:
        return None

    try:
        if os.path.exists(thumb_path) and os.path.getmtime(thumb_path) >= os.path.getmtime(image_path):
            return thumb_path
    except OSError:
        pass

    try:
        from PIL import Image

        target_dir = os.path.dirname(thumb_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            ratio = min(1.0, float(max_size) / float(max(width, height)))
            if ratio < 1.0:
                image = image.resize(
                    (max(1, int(width * ratio)), max(1, int(height * ratio))),
                    Image.LANCZOS,
                )
            # 先写临时文件再原子替换，避免并发请求读到半张图
            temp_path = f"{thumb_path}.tmp"
            image.save(temp_path, "JPEG", quality=quality, optimize=True)
            os.replace(temp_path, thumb_path)

        return thumb_path
    except Exception as exc:  # 缩略图失败不影响主流程
        log.warning("生成缩略图失败 %s: %s", image_path, exc)
        return None


__all__ = ["thumbnail_path_for", "ensure_thumbnail", "THUMB_SUFFIX"]
