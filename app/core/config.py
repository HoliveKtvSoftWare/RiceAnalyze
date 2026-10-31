"""Portable plan A configuration with stable model-list APIs."""
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.paths import BACKEND_ROOT, backend_path


class Settings(BaseSettings):
    BACKEND_ROOT: ClassVar[Path] = BACKEND_ROOT
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
        validate_default=True,
    )

    DATABASE_URL: str
    SECRET_KEY: str
    REDIS_URL: str = "redis://localhost:6379/0"
    STORAGE_PATH: str = "./app_storage/user_data"
    YOLO_MODEL_PATH: str = "./models_yolo/stem/yolov8s-1.pt"
    LEAF_MODEL_PATH: str = "./models_yolo/leaf/leaf-our-best.pt"
    DB_ECHO: bool = False
    # Compatibility mode: graph cut for sides, constrained contours for bodies.
    LEAF_POSTPROCESS: Literal["graphcut", "conservative", "off"] = "graphcut"

    @field_validator("STORAGE_PATH", "YOLO_MODEL_PATH", "LEAF_MODEL_PATH")
    @classmethod
    def resolve_project_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("路径不能为空")
        return backend_path(value)

    def get_models_map(self) -> dict:
        return {"yolov8s": self.YOLO_MODEL_PATH}

    def get_model_path(self, model_name: Optional[str] = None) -> str:
        return self.YOLO_MODEL_PATH

    def get_available_models(self) -> list:
        return [{"name": key, "path": value} for key, value in self.get_models_map().items()]

    def get_leaf_models_map(self) -> dict:
        # Only OUR is supported; obsolete LEAF_MODELS settings are ignored.
        return {"leaf-our": self.LEAF_MODEL_PATH}

    def resolve_leaf_model_key(self, model_name: Optional[str] = None) -> str:
        return "leaf-our"

    def get_leaf_model_path(self, model_name: Optional[str] = None) -> str:
        return self.LEAF_MODEL_PATH

    def get_leaf_available_models(self) -> list:
        return [{"name": key, "path": value} for key, value in self.get_leaf_models_map().items()]


settings = Settings()
