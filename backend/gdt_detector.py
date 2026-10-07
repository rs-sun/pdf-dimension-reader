"""
gdt_detector.py — YOLO-B GD&T 符号分类器（ONNX Runtime 推理）

职责边界：给定一张页面渲染图 + GD&T 框坐标列表，返回每个框的符号类别。
不碰 assembler 逻辑，不碰 OCR，不创建新框。
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image

import onnxruntime as ort

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "position", "perpendicularity", "parallelism", "coaxiality",
    "roundness", "cylindricity", "straightness", "angularity",
    "surface_profile", "line_profile", "flatness", "symmetry",
]

# GD&T 符号 Unicode 映射（前端显示用）
CLASS_SYMBOLS = {
    "position": "⊕", "perpendicularity": "⊥", "parallelism": "∥",
    "coaxiality": "◎", "roundness": "○", "cylindricity": "⌭",
    "straightness": "—", "angularity": "∠",
    "surface_profile": "⌓", "line_profile": "⌒",
    "flatness": "▱", "symmetry": "≡",
}


class GDTDetector:
    """单例模型加载，复用跨页面调用。"""

    def __init__(self, model_path=None, conf=0.3, crop_size=400):
        """
        Args:
            model_path: best.onnx 路径。默认自动定位到
                        backend/yolo_gdt/gdt_crop_v1/weights/best.onnx
            conf: 置信度阈值。默认 0.3（宁多检少漏，
                  误检会被线段网格匹配过滤掉）。
            crop_size: 裁切尺寸，必须与训练一致（400）。
        """
        if model_path is None:
            model_path = str(
                Path(__file__).parent / "yolo_gdt" / "gdt_crop_v1" / "weights" / "best.onnx"
            )

        # 优先 GPU，不可用则 fallback CPU
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )

        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.crop_size = crop_size
        provider = self.session.get_providers()[0]
        logger.info(f"[gdt_detector] ONNX 模型加载完成: {model_path}, conf={conf}, provider={provider}")

    def detect(self, page_image, gdt_frames):
        """
        对每个 GD&T 框裁切 crop_size×crop_size 区域，送 ONNX 推理。

        Args:
            page_image: PIL.Image (RGB) 或 PNG 文件路径 (str)
            gdt_frames: list of dict，每个 dict 至少包含:
                {
                    "bbox": [x1, y1, x2, y2],  # 像素坐标
                    ...（其他字段透传）
                }

        Returns:
            list of dict，与输入一一对应:
                {
                    "symbol_class": int,     # 0-11，未检出为 -1
                    "symbol_name": str,      # 如 "position"，未检出为 "unknown"
                    "symbol_unicode": str,   # 如 "⊕"，未检出为 "?"
                    "confidence": float,     # 0.0-1.0，未检出为 0.0
                }
        """
        if not gdt_frames:
            return []

        if isinstance(page_image, str):
            img = np.array(Image.open(page_image).convert("RGB"))
        else:
            img = np.array(page_image.convert("RGB"))
        h, w = img.shape[:2]
        results = []

        for frame in gdt_frames:
            bbox = frame["bbox"]  # [x1, y1, x2, y2] 像素坐标
            crop = self._crop_around_bbox(img, bbox, w, h)
            det = self._infer_single(crop)
            results.append(det)

        detected_count = sum(1 for r in results if r["symbol_class"] >= 0)
        logger.info(
            f"[gdt_detector] {len(gdt_frames)} 框, "
            f"{detected_count} 检出, "
            f"{len(gdt_frames) - detected_count} 未检出"
        )
        return results

    def _crop_around_bbox(self, img, bbox, img_w, img_h):
        """以 bbox 中心裁切 crop_size×crop_size，超出边界白色 padding。"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = self.crop_size // 2

        # 计算裁切区域（可能超出图片边界）
        crop_x1 = cx - half
        crop_y1 = cy - half
        crop_x2 = cx + half
        crop_y2 = cy + half

        # 创建白色画布
        canvas = np.full((self.crop_size, self.crop_size, 3), 255, dtype=np.uint8)

        # 计算实际可用的源区域和目标区域
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img_w, crop_x2)
        src_y2 = min(img_h, crop_y2)

        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        return canvas

    def _infer_single(self, crop_img):
        """对单个裁切图推理，返回最高置信度的检出。"""
        _UNKNOWN = {
            "symbol_class": -1,
            "symbol_name": "unknown",
            "symbol_unicode": "?",
            "confidence": 0.0,
        }

        # 预处理: 400×400 RGB uint8 → (1, 3, 640, 640) float32 [0-1]
        img_resized = np.array(Image.fromarray(crop_img).resize((640, 640)))
        input_tensor = img_resized.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC → CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, 640, 640)

        # 推理
        output = self.session.run(None, {self.input_name: input_tensor})
        # output[0] shape = (1, 16, 8400)
        predictions = output[0][0]  # (16, 8400)
        predictions = predictions.T  # (8400, 16)

        # 后处理: 前 4 列 = cx,cy,w,h; 后 12 列 = class scores
        class_scores = predictions[:, 4:]  # (8400, 12)
        max_scores = class_scores.max(axis=1)  # (8400,)

        # 取全局最高分
        best_idx = max_scores.argmax()
        best_conf = float(max_scores[best_idx])

        if best_conf < self.conf:
            return _UNKNOWN

        cls_id = int(class_scores[best_idx].argmax())

        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            logger.warning(f"[gdt_detector] 异常 class_id={cls_id}, 跳过")
            return _UNKNOWN

        name = CLASS_NAMES[cls_id]
        return {
            "symbol_class": cls_id,
            "symbol_name": name,
            "symbol_unicode": CLASS_SYMBOLS.get(name, "?"),
            "confidence": best_conf,
        }
