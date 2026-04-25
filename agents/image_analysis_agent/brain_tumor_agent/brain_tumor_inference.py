"""
Brain Tumor Detection Agent - Powered by heyi-Trans-master
==========================================================

本模块接入 ``heyi-Trans-master`` 通用视觉 Transformer 框架，完成
脑肿瘤分割 + 病灶标记 + AI 辅助诊断的三阶段流水线。

三阶段标准流程：
    1. segment_image(image_path)         -> 生成分割掩膜 (mask)
    2. mark_lesion(image_path, mask)     -> 在原图上叠加高亮病灶区域，输出标注图
    3. diagnose(image_path, mask)        -> 结合影像与掩膜给出 AI 辅助诊断文本

便捷方法：
    predict(image_path, output_path)     -> 串行执行上述三步并返回统一结果字典

返回值约定（predict）：
    {
        "status": "success" | "not_implemented" | "error",
        "mask_path":  Optional[str],      # 保留字段（当前未单独落盘）
        "marked_path": Optional[str],     # 病灶标记图路径
        "diagnosis": str,                 # AI 辅助诊断文本
        "stages": {
            "segmentation": bool,
            "lesion_marking": bool,
            "ai_diagnosis": bool,
        },
        "message": str,                   # 给前端展示的说明
    }

权重加载：
    ``model_path`` 指向的是一个 PyTorch state-dict 文件 (``.pth``)。
    文件不存在时适配器会自动回退到 *ImageNet 预训练 backbone + 未微调 head*
    的演示模式，诊断文本会明确提醒用户当前结论仅供流程验证。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BrainTumorAgent:
    """脑肿瘤检测智能体（基于 heyi-Trans-master ViT 骨干）。"""

    IMPLEMENTED: bool = True

    DEFAULT_OUTPUT_FILENAME = "brain_tumor_plot.png"

    def __init__(
        self,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Any = None,
    ):
        self.model_path = model_path
        self.output_dir = output_dir or "./uploads/brain_tumor_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = config
        self._adapter = None
        self._adapter_error: Optional[Exception] = None

    def _get_adapter(self):
        """按需懒加载 heyi 适配器，失败时缓存异常对象以便 diagnose 降级。"""
        if self._adapter is not None or self._adapter_error is not None:
            return self._adapter

        try:
            from ..heyi_adapter import HeyiVisionAdapter

            image_size = 224
            device = None
            if self.config is not None:
                image_size = int(getattr(self.config, "heyi_image_size", 224))
                device = getattr(self.config, "heyi_device", None)

            self._adapter = HeyiVisionAdapter.get(
                task="brain_tumor",
                model_path=self.model_path,
                image_size=image_size,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            self._adapter_error = e
            logger.exception("[BrainTumorAgent] 初始化 HeyiVisionAdapter 失败")
        return self._adapter

    def segment_image(self, image_path: str) -> Any:
        """第一阶段：对 MRI/CT 进行病灶分割，返回包含掩膜与原图的字典。"""
        adapter = self._get_adapter()
        if adapter is None:
            raise RuntimeError(
                f"脑肿瘤分割模型不可用：{self._adapter_error}"
            )
        mask, orig_rgb = adapter.infer_mask(image_path)
        return {"mask": mask, "orig_rgb": orig_rgb}

    def mark_lesion(
        self,
        image_path: str,
        mask: Any,
        output_path: Optional[str] = None,
    ) -> str:
        """第二阶段：将分割掩膜叠加到原图上，输出标注图路径。"""
        adapter = self._get_adapter()
        if adapter is None:
            raise RuntimeError(
                f"脑肿瘤分割模型不可用：{self._adapter_error}"
            )

        if isinstance(mask, dict):
            mask_np = mask["mask"]
            orig_rgb = mask.get("orig_rgb")
        else:
            mask_np = mask
            orig_rgb = None

        if orig_rgb is None:
            import numpy as np  # noqa: WPS433
            from PIL import Image  # noqa: WPS433

            orig_rgb = np.array(Image.open(image_path).convert("RGB"))

        target = output_path or os.path.join(self.output_dir, self.DEFAULT_OUTPUT_FILENAME)
        return adapter.overlay(
            orig_rgb,
            mask_np,
            target,
            color_rgb=(255, 64, 64),
        )

    def diagnose(self, image_path: str, mask: Any) -> str:
        """第三阶段：基于分割结果生成中文 AI 辅助诊断文本。"""
        adapter = self._get_adapter()
        if adapter is None:
            raise RuntimeError(
                f"脑肿瘤分割模型不可用：{self._adapter_error}"
            )

        mask_np = mask["mask"] if isinstance(mask, dict) else mask
        stats = adapter.mask_statistics(mask_np)

        area_px = stats["mask_area_px"]
        area_ratio_pct = stats["mask_area_ratio"] * 100
        num_regions = stats["num_regions"]
        weights_loaded = adapter.weights_loaded

        if area_px == 0:
            tail = "" if weights_loaded else (
                "\n\n⚠️ 当前运行在 *演示模式*（未加载微调权重），仅用于验证流水线，不能作为临床依据。"
            )
            return (
                "**AI 初步结论（脑肿瘤检测）**\n\n"
                "- 当前切片未检测到明显的肿瘤征象。\n"
                "- 建议结合其他切片以及多序列 MRI (T1 / T2 / FLAIR) 综合判读。"
                + tail
            )

        location = adapter.describe_location(stats, language="zh")
        if area_ratio_pct < 0.1:
            severity = "极小范围的可疑区域，假阳性可能性较高"
        elif area_ratio_pct < 1.5:
            severity = "小范围可疑区域"
        elif area_ratio_pct < 6:
            severity = "中等范围的可疑区域"
        else:
            severity = "较大范围的可疑区域"

        lines = [
            "**AI 初步结论（脑肿瘤检测）**",
            "",
            f"- 共检测到 **{num_regions}** 个可疑区域；主病灶位于 **{location}**。",
            f"- 病灶面积约 **{area_px}** 像素，占切片面积 **{area_ratio_pct:.2f}%**，属于 **{severity}**。",
        ]
        bbox = stats.get("bbox")
        if bbox is not None:
            x, y, bw, bh = bbox
            lines.append(f"- 主病灶外接矩形：x={x}, y={y}, w={bw}, h={bh}")

        if not weights_loaded:
            lines.append(
                "- ⚠️ 当前使用 *ImageNet 预训练 ViT 特征 + 未微调二值分割头* 的**演示模式**，"
                "结论仅用于验证前后端链路，请勿用于临床决策。"
            )

        lines.append("")
        lines.append(
            "**建议**：请神经外科 / 影像科医师结合临床症状、多模态 MRI 序列与随访影像综合评估；"
            "系统已默认进入人工复核流程。"
        )
        return "\n".join(lines)

    # ---------------------------------------------------------------
    # 下面的方法通常无需修改，组合调用上述三个阶段即可。
    # ---------------------------------------------------------------
    def predict(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            "message": "脑肿瘤检测算法尚未接入，当前仅为接口占位。",
        }

        if not self.IMPLEMENTED:
            return result

        # 在新一轮推理前先清理上一次的结果图，避免 segmentation 阶段就崩掉时
        # 前端仍然展示上一张图（FastAPI 端按 *文件存在* 判断是否附 result_image）。
        target_path = output_path or os.path.join(
            self.output_dir, self.DEFAULT_OUTPUT_FILENAME
        )
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
        except OSError as e:  # noqa: BLE001
            logger.warning(
                "[BrainTumorAgent] 清理旧结果图失败 (%s): %s", target_path, e
            )

        try:
            seg_bundle = self.segment_image(image_path)
            result["stages"]["segmentation"] = True

            marked_path = self.mark_lesion(image_path, seg_bundle, output_path=output_path)
            result["stages"]["lesion_marking"] = True
            result["marked_path"] = marked_path

            diagnosis = self.diagnose(image_path, seg_bundle)
            result["stages"]["ai_diagnosis"] = True
            result["diagnosis"] = diagnosis

            result["status"] = "success"
            adapter = self._get_adapter()
            if adapter is not None and adapter.weights_loaded:
                result["message"] = "脑肿瘤检测完成。"
            else:
                result["message"] = "脑肿瘤检测流水线已跑通（当前为演示模式，未加载微调权重）。"
        except NotImplementedError as e:
            result["status"] = "not_implemented"
            result["message"] = str(e)
        except Exception as e:  # noqa: BLE001
            logger.exception("BrainTumorAgent.predict failed")
            result["status"] = "error"
            result["message"] = f"脑肿瘤检测执行异常：{e}"

        return result
