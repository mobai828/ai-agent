"""
heyi-Trans-master 视觉模型适配层 (Heyi Vision Adapter)
=====================================================

把 `heyi-Trans-master` 项目中的通用视觉 Transformer 框架，桥接到主项目
(Multi-Agent Medical Assistant) 里 BrainTumorAgent / BrainStrokeAgent 的
三阶段流水线 (segmentation -> lesion_marking -> ai_diagnosis) 上。

为什么不直接使用 heyi 的 `UniversalVisionModel`？
----------------------------------------------
1. heyi 的 `heads/segmentation/segmentation_head.py` 使用了三级相对导入
   (`from ...architect...`)，要求 `heyi-Trans-master` 目录本身是一个 Python
   包才能工作，而当前仓库里它并没有顶层 ``__init__.py``。
2. heyi 自带的分割头是 FCN/Conv2d 结构，而 ViT 编码器输出的是序列式
   patch features (``[B, N, D]``)，二者形状不匹配。

折中方案：**只复用 heyi 的 ViT 骨干网络**（通过其 ``ComponentRegistry``
正式获取），再自行接一个轻量的二值分割解码器。这样：

- 真正用到了 heyi 的 ``architect.core.registry.ComponentRegistry`` 注册机制
- 真正用到了 heyi 的 ``backbone.encoders.vit_encoder.ViTEncoder`` 实现
- 不依赖 heyi 尚未跑通的 heads 组件
- 如果用户提供了微调过的权重文件，就直接加载并用于生产推理
- 如果没有权重（当前默认情况），则使用 ImageNet 预训练的 ViT 特征 +
  未训练分割头，以 *演示* 模式跑通整条链路（诊断文本中会明确提醒）
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HEYI_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "heyi-Trans-master")
)


def _ensure_heyi_on_path() -> None:
    if os.path.isdir(_HEYI_ROOT) and _HEYI_ROOT not in sys.path:
        sys.path.insert(0, _HEYI_ROOT)


class HeyiVisionAdapter:
    """基于 heyi ViT 骨干 + 轻量二值分割头的医疗影像分割适配器。"""

    _instances: Dict[str, "HeyiVisionAdapter"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        task: str = "brain_tumor",
        model_path: Optional[str] = None,
        image_size: int = 224,
        backbone_name: str = "vit_b_16",
        pretrained_backbone: bool = True,
        device: Optional[str] = None,
    ):
        _ensure_heyi_on_path()

        import torch  # noqa: WPS433 (延迟导入，避免 agentgithub 主流程加载时强依赖)
        import torchvision.transforms as T
        from PIL import Image

        import backbone.encoders  # noqa: F401  触发 ComponentRegistry 注册副作用
        from architect.core.registry import ComponentRegistry

        self.task = task
        self.image_size = image_size
        self.backbone_name = backbone_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        encoder_cfg = {
            "model_name": backbone_name,
            "pretrained": pretrained_backbone,
            "freeze_backbone": False,
        }
        self.encoder = ComponentRegistry.get_backbone("vit_encoder", encoder_cfg)
        self.encoder.eval()

        hidden = self.encoder.get_output_dim()
        patch_size = self.encoder.patch_size
        self.grid = image_size // patch_size

        self.decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden // 2, 2),
        )

        self._weights_loaded = self._try_load_weights(torch, model_path)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder.eval()

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._torch = torch
        self._Image = Image

    def _try_load_weights(self, torch_mod, model_path: Optional[str]) -> bool:
        if not model_path or not os.path.isfile(model_path):
            logger.info(
                "[HeyiAdapter:%s] 未提供微调权重 (path=%s)，回退到 ImageNet 预训练 backbone + 未微调 head 的演示模式。",
                self.task,
                model_path,
            )
            return False

        try:
            state = torch_mod.load(model_path, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[HeyiAdapter:%s] 权重加载失败 (%s)，回退到演示模式。",
                self.task,
                e,
            )
            return False

        try:
            if isinstance(state, dict) and "encoder" in state and "decoder" in state:
                self.encoder.load_state_dict(state["encoder"], strict=False)
                self.decoder.load_state_dict(state["decoder"], strict=False)
            elif isinstance(state, dict) and all(
                k.startswith(("0.", "1.", "2.", "3.", "4.")) for k in state.keys()
            ):
                self.decoder.load_state_dict(state, strict=False)
            elif isinstance(state, dict):
                enc_like = {
                    k[len("encoder.") :]: v
                    for k, v in state.items()
                    if k.startswith("encoder.")
                }
                dec_like = {
                    k[len("decoder.") :]: v
                    for k, v in state.items()
                    if k.startswith("decoder.")
                }
                if enc_like:
                    self.encoder.load_state_dict(enc_like, strict=False)
                if dec_like:
                    self.decoder.load_state_dict(dec_like, strict=False)
                if not (enc_like or dec_like):
                    self.encoder.load_state_dict(state, strict=False)
            logger.info("[HeyiAdapter:%s] 已加载微调权重: %s", self.task, model_path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[HeyiAdapter:%s] 权重格式不兼容 (%s)，回退到演示模式。",
                self.task,
                e,
            )
            return False

    @classmethod
    def get(cls, task: str, **kwargs: Any) -> "HeyiVisionAdapter":
        """任务粒度的单例缓存（brain_tumor / brain_stroke 各一份）。"""
        with cls._lock:
            if task not in cls._instances:
                cls._instances[task] = cls(task=task, **kwargs)
            return cls._instances[task]

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    def infer_mask(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """对图像执行分割。

        Returns:
            (mask, orig_rgb):
              - mask: ``np.uint8`` 形状 ``[H, W]``，二值 (0/1)
              - orig_rgb: ``np.uint8`` 形状 ``[H, W, 3]``，原图 RGB
        """
        torch = self._torch
        Image = self._Image

        pil = Image.open(image_path).convert("RGB")
        orig_rgb = np.array(pil)
        orig_h, orig_w = orig_rgb.shape[:2]

        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.encoder(x)
            patch_feats = feats["patch_features"]
            logits = self.decoder(patch_feats)
            prob_fg = torch.softmax(logits, dim=-1)[..., 1]
            prob_map = prob_fg.view(1, 1, self.grid, self.grid)
            prob_map = torch.nn.functional.interpolate(
                prob_map,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            prob_np = prob_map[0, 0].cpu().numpy()

        if self._weights_loaded:
            threshold = 0.5
        else:
            threshold = float(prob_np.mean() + prob_np.std())
            threshold = min(max(threshold, 0.55), 0.9)

        mask = (prob_np >= threshold).astype(np.uint8)
        return mask, orig_rgb

    def overlay(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        output_path: str,
        color_rgb: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.45,
        contour_bgr: Tuple[int, int, int] = (0, 255, 255),
        contour_thickness: int = 2,
    ) -> str:
        """把二值 mask 以半透明颜色叠加到原图并绘制轮廓，落盘返回路径。

        Args:
            contour_thickness: 轮廓粗细，默认 2；上层调用方可调大让标记更醒目。
        """
        import cv2

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        overlay_rgb = image_rgb.copy()
        color_layer = np.zeros_like(image_rgb)
        color_layer[..., 0] = color_rgb[0]
        color_layer[..., 1] = color_rgb[1]
        color_layer[..., 2] = color_rgb[2]

        mask_bool = mask.astype(bool)
        if mask_bool.any():
            overlay_rgb[mask_bool] = (
                overlay_rgb[mask_bool].astype(np.float32) * (1 - alpha)
                + color_layer[mask_bool].astype(np.float32) * alpha
            ).astype(np.uint8)

        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(
            overlay_bgr, contours, -1, contour_bgr, max(int(contour_thickness), 1)
        )
        cv2.imwrite(output_path, overlay_bgr)
        return output_path

    def mask_statistics(self, mask: np.ndarray) -> Dict[str, Any]:
        """从掩膜中提取面积 / 位置 / 形状特征，供 diagnose 使用。"""
        import cv2

        h, w = mask.shape
        area_px = int(mask.sum())
        total = h * w
        area_ratio = float(area_px) / float(total) if total > 0 else 0.0

        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        num_regions = len(contours)

        largest_area = 0
        centroid: Optional[Tuple[int, int]] = None
        bbox: Optional[Tuple[int, int, int, int]] = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest_area = int(cv2.contourArea(largest))
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
            x, y, bw, bh = cv2.boundingRect(largest)
            bbox = (int(x), int(y), int(bw), int(bh))

        return {
            "image_shape": (h, w),
            "mask_area_px": area_px,
            "mask_area_ratio": area_ratio,
            "num_regions": num_regions,
            "largest_region_area": largest_area,
            "centroid": centroid,
            "bbox": bbox,
        }

    @staticmethod
    def describe_location(stats: Dict[str, Any], language: str = "zh") -> str:
        """根据质心给出粗略的 "左/右 + 上/中/下" 位置描述。"""
        centroid = stats.get("centroid")
        if centroid is None:
            return "无法定位病灶中心" if language == "zh" else "unable to localize lesion"

        cx, cy = centroid
        h, w = stats["image_shape"]
        if language == "zh":
            horiz = "左" if cx < w * 0.4 else ("右" if cx > w * 0.6 else "中线")
            vert = "上部" if cy < h * 0.4 else ("下部" if cy > h * 0.6 else "中部")
            return f"{horiz}脑{vert}"

        horiz_en = "left" if cx < w * 0.4 else ("right" if cx > w * 0.6 else "midline")
        vert_en = (
            "superior" if cy < h * 0.4 else ("inferior" if cy > h * 0.6 else "middle")
        )
        return f"{horiz_en} {vert_en}"
