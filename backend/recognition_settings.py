"""按请求传递的识别后端选项。

不传 settings ⇒ from_request 返回 None ⇒ pipeline 走现有 Config 行为。
非法取值回落到 ocr_only（安全的老办法），并记进 invalid_fields 供响应标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIMENSION_PATHS = ("vector_only", "vector_plus_ocr", "ocr_only")
OCR_ENGINES = ("rapidocr", "paddle", "yolo_char")
VIEW_SEG_BACKENDS = ("yolo", "watershed", "minesweeper")
GDT_BACKENDS = ("hybrid", "vector", "yolo")
_LEGACY_DIMENSION_PATH = "ocr_only"
_OCR_ENGINE_TO_BACKEND = {"rapidocr": "rapidocr", "paddle": "paddleocr"}


@dataclass(frozen=True)
class RecognitionSettings:
    dimension_path: str = _LEGACY_DIMENSION_PATH
    ocr_engine: str | None = None
    view_seg: str | None = None
    gdt_backend: str | None = None
    invalid_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def is_vector_only(self) -> bool:
        """纯矢量、不跑 OCR：走 candidate_only 早返回缝。"""
        return self.dimension_path == "vector_only"

    @property
    def wants_ocr_merge(self) -> bool:
        """矢量+OCR：打开矢量消费合并（含 approval）。"""
        return self.dimension_path == "vector_plus_ocr"

    @property
    def requires_strict_yolo(self) -> bool:
        return self.is_vector_only

    def effective_ocr_engine(self) -> str | None:
        """菜单值转换为内部引擎名；缺省/占位值不设置覆盖。"""
        if self.is_vector_only:
            return None
        return _OCR_ENGINE_TO_BACKEND.get(self.ocr_engine or "")

    @classmethod
    def from_request(cls, data: dict[str, Any] | None) -> "RecognitionSettings | None":
        if not isinstance(data, dict):
            return None
        raw = data.get("settings")
        if not isinstance(raw, dict):
            return None
        invalid: dict[str, Any] = {}
        path = raw.get("dimension_path")
        if path in DIMENSION_PATHS:
            dimension_path = path
        else:
            dimension_path = _LEGACY_DIMENSION_PATH
            if path is not None:
                invalid["dimension_path"] = path

        engine = raw.get("ocr_engine")
        if engine in OCR_ENGINES:
            ocr_engine = engine
        else:
            ocr_engine = None
            if engine is not None:
                invalid["ocr_engine"] = engine

        view_seg_backend = raw.get("view_seg")
        if view_seg_backend in VIEW_SEG_BACKENDS:
            view_seg = view_seg_backend
        else:
            view_seg = None
            if view_seg_backend is not None:
                invalid["view_seg"] = view_seg_backend

        gdt_backend_raw = raw.get("gdt_backend")
        if gdt_backend_raw in GDT_BACKENDS:
            gdt_backend = gdt_backend_raw
        else:
            gdt_backend = None
            if gdt_backend_raw is not None:
                invalid["gdt_backend"] = gdt_backend_raw

        return cls(
            dimension_path=dimension_path,
            ocr_engine=ocr_engine,
            view_seg=view_seg,
            gdt_backend=gdt_backend,
            invalid_fields=invalid,
        )
