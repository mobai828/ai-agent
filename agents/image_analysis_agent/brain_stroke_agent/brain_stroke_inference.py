"""
Brain Stroke Detection Agent - Powered by heyi-Trans-master
===========================================================

本模块接入 ``heyi-Trans-master`` 通用视觉 Transformer 框架，完成脑卒中
梗死 / 出血灶分割 + 病灶标记 + AI 辅助诊断的三阶段流水线。

三阶段标准流程：
    1. segment_image(image_path)         -> 生成分割掩膜 (mask)
    2. mark_lesion(image_path, mask)     -> 在原图上叠加高亮病灶区域，输出标注图
    3. diagnose(image_path, mask)        -> 结合影像与掩膜给出 AI 辅助诊断文本

分割推理优先走 **同学部署的远程分割 API**（v2.0，默认
http://222.198.105.83:8000），拿到真实训练过的 mask；远程不可达或失败时
自动降级到本地 ``HeyiVisionAdapter`` 的演示模式，保证链路不中断。
返回值约定与 BrainTumorAgent 保持一致，详见 brain_tumor_inference.py。

关于"远程探活"的可恢复性
-------------------------
``BrainStrokeAgent`` 由 ``ImageAnalysisAgent`` 持有，本身是单例。早期实现
里只要 ``/segment`` 偶发失败一次就把 ``_remote_alive`` 永久置 ``False``，
之后所有上传都默默落到本地演示模式 —— 用户看到的就是叠了一堆"明显
不是真实分割"的蓝色色块（横跨颅骨外的黑色背景）。

修复策略：
    * ``/health`` 检查通过后视为 *服务可用*；
    * 单次 ``/segment`` 失败只标记当前请求降级，不触碰 ``_remote_alive``，
      这样下一次上传仍会再试一次远程；
    * 引入 ``_remote_cooldown_until`` 时间戳：连续多次失败时给一个 60s
      的冷却窗（避免反复打挂的服务），冷却结束自动重新探活。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 连续失败到达该阈值后进入冷却窗（避免每次请求都打 N 秒的远程超时）
_REMOTE_FAILURE_THRESHOLD = 3
_REMOTE_COOLDOWN_SECONDS = 60.0


class BrainStrokeAgent:
    """脑卒中检测智能体。

    分割策略（按优先级）：
      1. **远程** Heyi 分割服务 (``HEYI_REMOTE_URL``)，对应同学部署的
         训练好的脑卒中模型；
      2. **本地** ``HeyiVisionAdapter``（ImageNet 预训练 ViT + 未微调
         二值分割头，仅用于演示和降级）。

    任务粒度：
      - ``task="auto"``        让远程服务自动判定（仅作为兜底）
      - ``task="hemorrhage"``  显式指定为出血灶分割
      - ``task="ischemia"``    显式指定为缺血灶分割

    上层（Web 调用方 / 其他模块）应当 *显式* 传入 ``hemorrhage`` 或
    ``ischemia``，因为出血灶与缺血灶在影像学上差异显著，由上游分类器
    给出明确指令能让分割模型选对最合适的权重，提高准确率。
    """

    IMPLEMENTED: bool = True
    SUPPORTED_TASKS = ("auto", "hemorrhage", "ischemia")

    DEFAULT_OUTPUT_FILENAME = "brain_stroke_plot.png"

    def __init__(
        self,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Any = None,
    ):
        self.model_path = model_path
        self.output_dir = output_dir or "./uploads/brain_stroke_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = config
        self._adapter = None
        self._adapter_error: Optional[Exception] = None

        self._remote_client = None
        self._remote_checked = False
        self._remote_alive = False
        # 连续失败计数 + 冷却窗口；用于把"一次抖动"和"服务真挂了"区分开
        self._remote_consecutive_failures = 0
        self._remote_cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Adapter / Remote client lazy init
    # ------------------------------------------------------------------
    def _get_adapter(self):
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
                task="brain_stroke",
                model_path=self.model_path,
                image_size=image_size,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            self._adapter_error = e
            logger.exception("[BrainStrokeAgent] 初始化 HeyiVisionAdapter 失败")
        return self._adapter

    def _get_remote_client(self):
        """惰性创建远程客户端；配置未启用或未配 URL 时返回 None。"""
        if self._remote_client is not None:
            return self._remote_client
        if self.config is None:
            return None

        enabled = getattr(self.config, "heyi_remote_enabled", False)
        base_url = getattr(self.config, "heyi_remote_url", "") or ""
        if not enabled or not base_url:
            return None

        try:
            from ..heyi_remote_client import HeyiRemoteClient

            self._remote_client = HeyiRemoteClient.get(
                base_url=base_url,
                timeout=float(getattr(self.config, "heyi_remote_timeout", 120.0)),
                health_timeout=float(
                    getattr(self.config, "heyi_remote_health_timeout", 3.0)
                ),
                default_task=str(
                    getattr(self.config, "heyi_remote_task", "auto") or "auto"
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("[BrainStrokeAgent] 初始化远程客户端失败")
            self._remote_client = None
        return self._remote_client

    def _normalize_task(self, task: Optional[str]) -> str:
        """把上游传进来的 task 归一化成远程服务支持的取值。"""
        if not task:
            return "auto"
        cleaned = str(task).strip().lower()
        # 给前端可能传过来的中文 / 缩写做个兜底映射
        aliases = {
            "出血": "hemorrhage",
            "出血灶": "hemorrhage",
            "hemorrhagic": "hemorrhage",
            "haemorrhage": "hemorrhage",
            "缺血": "ischemia",
            "缺血灶": "ischemia",
            "ischemic": "ischemia",
            "ischaemia": "ischemia",
            "infarction": "ischemia",
            "梗死": "ischemia",
        }
        cleaned = aliases.get(cleaned, cleaned)
        if cleaned not in self.SUPPORTED_TASKS:
            logger.warning(
                "[BrainStrokeAgent] 不支持的 task=%r，回退到 'auto'。", task
            )
            return "auto"
        return cleaned

    def _refresh_remote_health(self) -> bool:
        """刷新远程服务可用性。

        流程：
          1. 没有 client / 配置关闭 → 永远 False
          2. 处于冷却窗内 → 直接返回 False，不打健康检查
          3. 否则调一次 ``/health``：成功置 alive=True / 清零失败计数；
             失败置 alive=False。
        """
        client = self._get_remote_client()
        if client is None:
            return False

        now = time.monotonic()
        if self._remote_cooldown_until > now:
            return False

        alive = client.is_alive()
        self._remote_alive = alive
        self._remote_checked = True
        if alive:
            self._remote_consecutive_failures = 0
            self._remote_cooldown_until = 0.0
        else:
            logger.warning(
                "[BrainStrokeAgent] 远程分割服务 %s /health 不可达，"
                "本次请求走本地 adapter。",
                client.base_url,
            )
        return alive

    def _record_remote_failure(self, scope: str, error: Exception) -> None:
        """记一次远程调用失败：累加计数，到阈值就开冷却窗。"""
        self._remote_consecutive_failures += 1
        if self._remote_consecutive_failures >= _REMOTE_FAILURE_THRESHOLD:
            self._remote_alive = False
            self._remote_cooldown_until = (
                time.monotonic() + _REMOTE_COOLDOWN_SECONDS
            )
            logger.warning(
                "[BrainStrokeAgent] 远程 %s 连续失败 %d 次 (最近一次: %s)，"
                "进入 %d 秒冷却，期间所有请求走本地 fallback。",
                scope,
                self._remote_consecutive_failures,
                error,
                int(_REMOTE_COOLDOWN_SECONDS),
            )
        else:
            logger.warning(
                "[BrainStrokeAgent] 远程 %s 单次失败 (累计 %d/%d): %s",
                scope,
                self._remote_consecutive_failures,
                _REMOTE_FAILURE_THRESHOLD,
                error,
            )

    def _try_remote_segment(
        self, image_path: str, task: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """尝试通过远程服务拿到 mask + 预览图；任何失败都返回 None 触发本地降级。

        策略：
          1. ``/segment`` 取 NIfTI mask -> 用于本地统计 / 诊断文本（必需）
          2. ``/segment/preview`` 取烤好的可视化 PNG -> 直接作为前端展示的
             *分割结果* 图（强烈推荐：服务端的可视化效果就是训练好的模型该有
             的样子，无需本地再叠加一遍）。预览图调用失败不致命，会回退到
             本地 overlay 重建。

        Args:
            task: ``"auto"`` / ``"hemorrhage"`` / ``"ischemia"``；
                  None 时使用客户端 ``default_task``（来自 config）。
        """
        client = self._get_remote_client()
        if client is None:
            return None

        # 冷却窗内直接走本地，不打远程
        if self._remote_cooldown_until > time.monotonic():
            return None
        # 第一次或之前不可达：重新探活
        if not self._remote_checked or not self._remote_alive:
            if not self._refresh_remote_health():
                return None

        try:
            mask, orig_rgb = client.segment(image_path, task=task)
        except Exception as e:  # noqa: BLE001
            self._record_remote_failure(f"/segment (task={task})", e)
            return None

        # /segment 成功就视为远程"基本可用"，清零失败计数
        self._remote_consecutive_failures = 0
        self._remote_cooldown_until = 0.0
        self._remote_alive = True

        # 预览图是 *锦上添花*：失败不会让 _remote_alive 变 False，
        # 只是这次 mark_lesion 用本地 overlay 重画。也不计入 failure_count。
        preview_png: Optional[bytes] = None
        try:
            preview_png = client.segment_preview(image_path, task=task)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[BrainStrokeAgent] 远程 /segment/preview (task=%s) 失败 (%s)，"
                "本次结果图将由本地 overlay 重新绘制 (远程 mask 仍然可用)。",
                task,
                e,
            )

        return {
            "mask": mask,
            "orig_rgb": orig_rgb,
            "source": "remote",
            "preview_png": preview_png,
            "task": task or "auto",
        }

    # ------------------------------------------------------------------
    # Three-stage pipeline
    # ------------------------------------------------------------------
    def segment_image(
        self, image_path: str, task: Optional[str] = None
    ) -> Dict[str, Any]:
        normalized_task = self._normalize_task(task)
        remote_result = self._try_remote_segment(image_path, task=normalized_task)
        if remote_result is not None:
            logger.info(
                "[BrainStrokeAgent] 使用远程 Heyi 服务的分割结果 (task=%s)。",
                normalized_task,
            )
            bundle = remote_result
        else:
            adapter = self._get_adapter()
            if adapter is None:
                raise RuntimeError(
                    f"脑卒中分割模型不可用（远程与本地均失败）：{self._adapter_error}"
                )
            mask, orig_rgb = adapter.infer_mask(image_path)
            bundle = {
                "mask": mask,
                "orig_rgb": orig_rgb,
                "source": "local",
                "task": normalized_task,
            }

        # 后处理：清洗噪点 + 形态学闭运算，让"病灶标记"和"诊断统计"基于
        # 视觉上更聚焦的主病灶集合；原始 mask 保留在 raw_mask 字段供调试。
        # 注意：mask 清洗仅影响 *诊断文本* 的统计；远程预览图(preview_png)
        # 是服务端直接出的可视化，不受这里清洗的影响。
        raw_mask = bundle["mask"]
        cleaned_mask = _clean_mask_for_visualization(raw_mask)
        if int(cleaned_mask.sum()) != int(raw_mask.sum()):
            logger.info(
                "[BrainStrokeAgent] 病灶 mask 已清洗噪点：raw_fg_px=%d -> cleaned_fg_px=%d",
                int(raw_mask.sum()),
                int(cleaned_mask.sum()),
            )
        bundle["mask"] = cleaned_mask
        bundle["raw_mask"] = raw_mask
        return bundle

    def mark_lesion(
        self,
        image_path: str,
        mask: Any,
        output_path: Optional[str] = None,
    ) -> str:
        if isinstance(mask, dict):
            mask_np = mask["mask"]
            orig_rgb = mask.get("orig_rgb")
            preview_png = mask.get("preview_png")
        else:
            mask_np = mask
            orig_rgb = None
            preview_png = None

        target = output_path or os.path.join(self.output_dir, self.DEFAULT_OUTPUT_FILENAME)

        # 优先策略：直接落盘服务端 /segment/preview 返回的 PNG。
        # 这样 "分割结果" 视觉效果与服务端训练好的模型保持完全一致，
        # 不会再因为本地 overlay/阈值/清洗参数差异而走样。
        if preview_png:
            try:
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                with open(target, "wb") as f:
                    f.write(preview_png)
                logger.info(
                    "[BrainStrokeAgent] 直接使用 /segment/preview 结果图: %s (bytes=%d)",
                    target,
                    len(preview_png),
                )
                return target
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[BrainStrokeAgent] 写入远程 preview PNG 失败 (%s)，"
                    "回退到本地 overlay 生成结果图。",
                    e,
                )

        if orig_rgb is None:
            import numpy as np  # noqa: WPS433
            from PIL import Image  # noqa: WPS433

            orig_rgb = np.array(Image.open(image_path).convert("RGB"))

        # 视觉参数：填充 + 较粗轮廓，避免标记被淹没在原图灰度里
        overlay_kwargs = dict(
            color_rgb=(64, 128, 255),       # 半透明蓝色填充
            alpha=0.55,                      # 填充更明显
            contour_bgr=(0, 220, 255),       # 黄色轮廓
            contour_thickness=3,             # 轮廓加粗，远观即可识别
        )

        adapter = self._get_adapter()
        if adapter is not None:
            return adapter.overlay(orig_rgb, mask_np, target, **overlay_kwargs)

        logger.info(
            "[BrainStrokeAgent] 本地 adapter 不可用，使用内置 overlay 工具继续输出标注图。"
        )
        return _fallback_overlay(orig_rgb, mask_np, target, **overlay_kwargs)

    def diagnose(self, image_path: str, mask: Any) -> str:
        if isinstance(mask, dict):
            mask_np = mask["mask"]
            source = mask.get("source", "local")
            task = mask.get("task", "auto")
        else:
            mask_np = mask
            source = "local"
            task = "auto"

        stats = self._compute_mask_stats(mask_np)

        area_px = stats["mask_area_px"]
        area_ratio_pct = stats["mask_area_ratio"] * 100
        num_regions = stats["num_regions"]

        is_remote = source == "remote"
        if is_remote:
            weights_loaded = True
            mode_tag = "remote"
        else:
            adapter = self._get_adapter()
            weights_loaded = bool(adapter and adapter.weights_loaded)
            mode_tag = "local" if adapter is not None else "unavailable"

        if area_px == 0:
            tail = ""
            if not weights_loaded and mode_tag != "remote":
                tail = (
                    "\n\n⚠️ 当前运行在 *演示模式*（未加载微调权重），仅用于验证流水线，不能作为临床依据。"
                )
            return (
                "**AI 初步结论（脑卒中检测）**\n\n"
                "- 当前切片未检测到明显的梗死 / 出血灶征象。\n"
                "- 建议结合 DWI、ADC 等多模态序列及临床时间窗综合判读。"
                + tail
            )

        location = self._describe_location(stats)
        if area_ratio_pct < 0.1:
            severity = "极小范围可疑信号（假阳性可能性较高）"
        elif area_ratio_pct < 1.5:
            severity = "小范围可疑梗死 / 出血灶"
        elif area_ratio_pct < 6:
            severity = "中等范围病灶"
        else:
            severity = "较大范围病灶"

        task_zh_map = {
            "hemorrhage": "出血灶（hemorrhage）",
            "ischemia": "缺血灶（ischemia）",
            "auto": "自动判定（auto）",
        }
        task_zh = task_zh_map.get(task, "自动判定（auto）")

        lines = [
            "**AI 初步结论（脑卒中检测）**",
            "",
            f"- 任务类型：**{task_zh}**",
            f"- 共检测到 **{num_regions}** 个可疑区域；主病灶位于 **{location}**。",
            f"- 病灶面积约 **{area_px}** 像素，占切片面积 **{area_ratio_pct:.2f}%**，属于 **{severity}**。",
        ]
        bbox = stats.get("bbox")
        if bbox is not None:
            x, y, bw, bh = bbox
            lines.append(f"- 主病灶外接矩形：x={x}, y={y}, w={bw}, h={bh}")

        if is_remote:
            lines.append(
                "- ✅ 本次分割由 **远程 Heyi 分割服务 (v2.0)** 完成，模型权重已经过脑卒中数据集训练。"
            )
        elif not weights_loaded:
            lines.append(
                "- ⚠️ 当前使用 *ImageNet 预训练 ViT 特征 + 未微调二值分割头* 的**演示模式**，"
                "结论仅用于验证前后端链路，请勿用于临床决策。"
            )

        lines.append("")
        lines.append(
            "**建议**：请神经内科 / 影像科医师结合发病时间窗、DWI-ADC 匹配关系、"
            "临床体征与随访影像综合评估；系统已默认进入人工复核流程。"
        )
        return "\n".join(lines)

    def predict(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """对脑部影像执行三阶段流水线。

        Args:
            image_path: 输入图像本地路径
            output_path: 结果图落盘路径（默认写到 ``output_dir/brain_stroke_plot.png``）
            task: 卒中亚型选择，``"auto"`` / ``"hemorrhage"`` / ``"ischemia"``。
                  上游（其他模块 / 前端 UI）应当显式给值，避免远程服务在
                  ``auto`` 模式下做次优判断。
        """
        normalized_task = self._normalize_task(task)
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
            "source": None,
            "task": normalized_task,
            "message": "脑卒中检测算法尚未接入，当前仅为接口占位。",
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
                "[BrainStrokeAgent] 清理旧结果图失败 (%s): %s", target_path, e
            )

        try:
            seg_bundle = self.segment_image(image_path, task=normalized_task)
            result["stages"]["segmentation"] = True
            result["source"] = seg_bundle.get("source", "local")
            # 真正落地的 task（远程默认值可能与上游传入不同；这里以执行结果为准）
            result["task"] = seg_bundle.get("task", normalized_task)

            marked_path = self.mark_lesion(image_path, seg_bundle, output_path=output_path)
            result["stages"]["lesion_marking"] = True
            result["marked_path"] = marked_path

            diagnosis = self.diagnose(image_path, seg_bundle)
            result["stages"]["ai_diagnosis"] = True
            result["diagnosis"] = diagnosis

            result["status"] = "success"
            if result["source"] == "remote":
                result["message"] = (
                    f"脑卒中检测完成（远程 Heyi 分割服务，task={result['task']}）。"
                )
            else:
                adapter = self._get_adapter()
                if adapter is not None and adapter.weights_loaded:
                    result["message"] = "脑卒中检测完成（本地微调权重）。"
                else:
                    result["message"] = (
                        "脑卒中检测流水线已跑通（当前为本地演示模式，未加载微调权重）。"
                    )
        except NotImplementedError as e:
            result["status"] = "not_implemented"
            result["message"] = str(e)
        except Exception as e:  # noqa: BLE001
            logger.exception("BrainStrokeAgent.predict failed")
            result["status"] = "error"
            result["message"] = f"脑卒中检测执行异常：{e}"

        return result

    # ------------------------------------------------------------------
    # Stats helpers (works for both remote & local mask sources)
    # ------------------------------------------------------------------
    def _compute_mask_stats(self, mask: Any) -> Dict[str, Any]:
        adapter = self._get_adapter()
        if adapter is not None:
            return adapter.mask_statistics(mask)
        return _fallback_mask_statistics(mask)

    def _describe_location(self, stats: Dict[str, Any]) -> str:
        adapter = self._get_adapter()
        if adapter is not None:
            return adapter.describe_location(stats, language="zh")
        return _fallback_describe_location(stats)


# ----------------------------------------------------------------------
# Pure numpy/cv2 fallbacks
# ----------------------------------------------------------------------
# 当本地 adapter 初始化失败（例如远程服务正常但本地没有 torch）时，仍需要
# 一条"能画图、能统计"的最小链路，保证 mark_lesion / diagnose 不至于崩掉。


def _fallback_overlay(
    image_rgb,
    mask,
    output_path: str,
    color_rgb=(64, 128, 255),
    alpha: float = 0.45,
    contour_bgr=(0, 255, 255),
    contour_thickness: int = 2,
) -> str:
    import cv2  # noqa: WPS433
    import numpy as np  # noqa: WPS433

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
    cv2.drawContours(overlay_bgr, contours, -1, contour_bgr, max(int(contour_thickness), 1))
    cv2.imwrite(output_path, overlay_bgr)
    return output_path


def _clean_mask_for_visualization(
    mask,
    min_relative_area: float = 0.0005,
    min_relative_to_max: float = 0.15,
):
    """清理分割 mask 中的噪点与碎片，让标记图更聚焦于主病灶。

    步骤：
      1. **形态学闭运算**（5x5 椭圆 kernel）合并相邻碎片、平滑边缘。
      2. **连通域过滤**：丢弃同时满足以下两条的小斑点：
         - 绝对面积 < 整图的 ``min_relative_area``（默认 0.05%）
         - 相对面积 < 最大病灶的 ``min_relative_to_max``（默认 15%）

    设计取舍：
      - 不强行只留 top-1 连通域：脑卒中可能是多发病灶，相对阈值能保留次要主灶。
      - 阈值偏宽松：宁可多留一两个偏小的，也不要把真实病灶过滤掉。
      - 输入若是空 mask 直接原样返回，节省一次 morph 调用。
    """
    import cv2  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    if mask is None:
        return mask
    arr = np.asarray(mask)
    if arr.ndim != 2 or int(arr.sum()) == 0:
        return arr.astype(np.uint8) if arr.ndim == 2 else arr

    h, w = arr.shape
    total = h * w

    bin8 = (arr > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(bin8, cv2.MORPH_CLOSE, kernel)
    closed_bin = (closed > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_bin, connectivity=8
    )
    if n_labels <= 1:
        return closed_bin

    fg_areas = stats[1:, cv2.CC_STAT_AREA]
    if fg_areas.size == 0:
        return closed_bin
    max_area = int(fg_areas.max())

    abs_threshold = max(int(total * min_relative_area), 4)
    rel_threshold = max(int(max_area * min_relative_to_max), 1)
    keep_threshold = max(abs_threshold, rel_threshold)

    cleaned = np.zeros_like(closed_bin)
    for idx in range(1, n_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= keep_threshold:
            cleaned[labels == idx] = 1
    return cleaned


def _fallback_mask_statistics(mask) -> Dict[str, Any]:
    import cv2  # noqa: WPS433

    h, w = mask.shape
    area_px = int(mask.sum())
    total = h * w
    area_ratio = float(area_px) / float(total) if total > 0 else 0.0

    contours, _ = cv2.findContours(
        (mask * 255).astype("uint8"),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    num_regions = len(contours)

    largest_area = 0
    centroid = None
    bbox = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_area = int(cv2.contourArea(largest))
        M = cv2.moments(largest)
        if M["m00"] > 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
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


def _fallback_describe_location(stats: Dict[str, Any]) -> str:
    centroid = stats.get("centroid")
    if centroid is None:
        return "无法定位病灶中心"
    cx, cy = centroid
    h, w = stats["image_shape"]
    horiz = "左" if cx < w * 0.4 else ("右" if cx > w * 0.6 else "中线")
    vert = "上部" if cy < h * 0.4 else ("下部" if cy > h * 0.6 else "中部")
    return f"{horiz}脑{vert}"
