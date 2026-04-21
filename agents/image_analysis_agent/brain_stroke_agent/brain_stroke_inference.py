"""
Brain Stroke Detection Agent - Reserved Interface (Stub)
=========================================================

本模块为脑卒中检测智能体的预留接口占位实现，后续由算法同学接入。

三阶段标准流程：
    1. segment_image(image_path)         -> 生成分割掩膜 (mask)
    2. mark_lesion(image_path, mask)     -> 在原图上叠加高亮病灶区域，输出标注图
    3. diagnose(image_path, mask)        -> 结合影像与掩膜给出 AI 辅助诊断文本

返回值约定与 BrainTumorAgent 保持一致，详见 brain_tumor_inference.py。
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BrainStrokeAgent:
    """脑卒中检测智能体（预留接口）。"""

    IMPLEMENTED: bool = False  # 算法接入完成后请将此标志改为 True

    def __init__(self, model_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.model_path = model_path
        self.output_dir = output_dir or "./uploads/brain_stroke_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def segment_image(self, image_path: str) -> Any:
        """第一阶段：对脑部影像进行梗死/出血灶分割，返回掩膜对象。"""
        raise NotImplementedError(
            "BrainStrokeAgent.segment_image 尚未实现。请接入脑卒中分割模型。"
        )

    def mark_lesion(self, image_path: str, mask: Any, output_path: Optional[str] = None) -> str:
        """第二阶段：将分割掩膜以高亮/轮廓叠加到原图，输出标注图路径。"""
        raise NotImplementedError(
            "BrainStrokeAgent.mark_lesion 尚未实现。请实现在原图上标记病灶区域的逻辑。"
        )

    def diagnose(self, image_path: str, mask: Any) -> str:
        """第三阶段：基于影像与分割掩膜，给出 AI 辅助诊断建议文本。"""
        raise NotImplementedError(
            "BrainStrokeAgent.diagnose 尚未实现。请接入 AI 辅助诊断模型。"
        )

    def predict(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """串行执行三个阶段，返回结构化结果。"""
        result: Dict[str, Any] = {
            "status": "not_implemented",
            "mask_path": None,
            "marked_path": None,
            "diagnosis": "",
            "stages": {
                "segmentation": False,
                "lesion_marking": False,
                "ai_diagnosis": False,
            },
            "message": "脑卒中检测算法尚未接入，当前仅为接口占位。",
        }

        if not self.IMPLEMENTED:
            return result

        try:
            mask = self.segment_image(image_path)
            result["stages"]["segmentation"] = True

            marked_path = self.mark_lesion(image_path, mask, output_path=output_path)
            result["stages"]["lesion_marking"] = True
            result["marked_path"] = marked_path

            diagnosis = self.diagnose(image_path, mask)
            result["stages"]["ai_diagnosis"] = True
            result["diagnosis"] = diagnosis

            result["status"] = "success"
            result["message"] = "脑卒中检测完成。"
        except NotImplementedError as e:
            result["status"] = "not_implemented"
            result["message"] = str(e)
        except Exception as e:
            logger.exception("BrainStrokeAgent.predict failed")
            result["status"] = "error"
            result["message"] = f"脑卒中检测执行异常：{e}"

        return result
