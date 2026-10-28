# 核心配置

import json
import logging
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _parse_models(raw: str, default_path: str) -> tuple:
    """
    解析 YOLO_MODELS 配置（结果按原始字符串缓存）。

    /api/analysis/models 与每次上传都会调用该配置，
    之前每次都要重新 json.loads 并构造新结构，属于不必要的重复开销。
    返回值是 tuple，避免调用方意外修改缓存内容。
    """
    try:
        models = json.loads(raw)
        if not isinstance(models, dict):
            models = {}
    except (json.JSONDecodeError, TypeError):
        log.warning("YOLO_MODELS 解析失败，回退到默认模型。")
        models = {}

    if not models:
        models = {"default": default_path}

    return tuple(models.items())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    STORAGE_PATH: str
    YOLO_MODEL_PATH: str = "./models_yolo/yolov8s.pt"
    YOLO_MODELS: str = '{"yolov8s": "./models_yolo/yolov8s.pt"}'

    # 剑叶（flag leaf）模型注册表，结构与 YOLO_MODELS 完全一致。
    # 模型文件尚未部署时不影响服务启动：上传剑叶图片会记录一条 failed 记录，
    # error_message 写明“模型文件不存在”，前端可直接展示。
    LEAF_MODEL_PATH: str = "./models_yolo/leaf.pt"
    LEAF_MODELS: str = '{"leaf-default": "./models_yolo/leaf.pt"}'

    def get_models_map(self) -> dict:
        """返回 {模型名: 模型路径} 字典（每次返回新的 dict，可安全修改）。"""
        return dict(_parse_models(self.YOLO_MODELS, self.YOLO_MODEL_PATH))

    def get_model_path(self, model_name: Optional[str]) -> str:
        models = dict(_parse_models(self.YOLO_MODELS, self.YOLO_MODEL_PATH))
        if model_name and model_name in models:
            return models[model_name]
        log.warning(f"未知模型 '{model_name}'，回退到默认模型 {self.YOLO_MODEL_PATH}")
        return self.YOLO_MODEL_PATH

    def get_available_models(self) -> list:
        return [
            {"name": name, "path": path}
            for name, path in _parse_models(self.YOLO_MODELS, self.YOLO_MODEL_PATH)
        ]

    # ---- 剑叶 ----
    def get_leaf_models_map(self) -> dict:
        """返回剑叶 {模型名: 模型路径} 字典。"""
        return dict(_parse_models(self.LEAF_MODELS, self.LEAF_MODEL_PATH))

    def get_leaf_model_path(self, model_name: Optional[str]) -> str:
        models = dict(_parse_models(self.LEAF_MODELS, self.LEAF_MODEL_PATH))
        key = self.resolve_leaf_model_key(model_name)
        return models.get(key, self.LEAF_MODEL_PATH)

    def resolve_leaf_model_key(self, model_name: Optional[str]) -> str:
        """
        把前端传来的剑叶模型标识归一化成 ``LEAF_MODELS`` 的 key。

        之所以需要它：前端页面可能在改 `.env` 之前就打开了，下拉框里还留着旧 key
        （例如 ``leaf-default``），直接拿旧值上来会 400。这里按以下顺序解析：

        1. 精确匹配 key；
        2. key 大小写不敏感匹配；
        3. 允许传权重文件名 / 路径（``leaf-our.pt``、``./models_yolo/leaf-our.pt``）；
        4. 仍匹配不上则回退到注册表第一项（默认模型），并打一条 warning。

        这样无论前端传 key、文件名还是过期名字，上传都不会因为“未知模型”被拒绝。
        """
        models = dict(_parse_models(self.LEAF_MODELS, self.LEAF_MODEL_PATH))
        if not models:
            return "leaf-default"

        default_key = next(iter(models))
        if not model_name:
            return default_key

        raw = str(model_name).strip()
        if raw in models:
            return raw

        lowered = raw.lower()
        for key in models:
            if key.lower() == lowered:
                return key

        basename = os.path.splitext(os.path.basename(raw.replace("\\", "/")))[0].lower()
        for key, path in models.items():
            if key.lower() == basename:
                return key
            path_basename = os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0].lower()
            if path_basename == basename:
                return key

        log.warning(
            "剑叶模型 '%s' 不在 LEAF_MODELS 中，已回退到默认模型 '%s'（可用: %s）",
            model_name, default_key, list(models.keys())
        )
        return default_key

    def get_leaf_available_models(self) -> list:
        return [
            {"name": name, "path": path}
            for name, path in _parse_models(self.LEAF_MODELS, self.LEAF_MODEL_PATH)
        ]


settings = Settings()
