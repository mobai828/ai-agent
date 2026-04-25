"""
Heyi 远程分割服务客户端 (Remote Heyi Stroke Segmentation Client)
================================================================

对接同学部署的 **医疗图像分割 API v2.0** (默认 http://222.198.105.83:8000)。

设计约束
--------
- **只负责"把模型推理这一段外包出去"**：
    * ``/segment``         -> 拿 NIfTI mask，解析成 2D ``np.uint8`` 二值矩阵
                              (用于本地 ``mask_statistics`` / 诊断文本生成)
    * ``/segment/preview`` -> 直接拿到服务端"烤"好的可视化 PNG 字节流
                              (用于前端展示的 *分割结果* 图像)
- **不承担 overlay / 统计 / 诊断文本生成**：这些都继续交给本地
  ``HeyiVisionAdapter``（只是它这时拿到的 mask 已经是真模型的结果而不是
  演示模式输出）。
- **故障一律抛 ``HeyiRemoteError``**，由上层 ``BrainStrokeAgent`` 捕获后
  降级到本地 adapter，确保业务永不中断。
- ``/segment`` 与 ``/segment/preview`` 是两种互补输出：前者保留二值矩阵
  供下游统计；后者直接给前端展示，无需本地再叠加。
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HeyiRemoteError(RuntimeError):
    """远程 Heyi 服务调用失败，调用方应据此进入本地降级。"""


class HeyiRemoteClient:
    """对接 Heyi 分割 API 的轻量 HTTP 客户端（线程安全）。

    典型用法::

        client = HeyiRemoteClient.get("http://222.198.105.83:8000")
        if client.is_alive():
            mask, orig_rgb = client.segment("ct.png", task="auto")
    """

    _instances: Dict[str, "HeyiRemoteClient"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str = "http://222.198.105.83:8000",
        timeout: float = 120.0,
        health_timeout: float = 3.0,
        default_task: str = "auto",
    ):
        import requests  # noqa: WPS433

        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.default_task = default_task
        self._session = requests.Session()
        self._requests = requests

    @classmethod
    def get(
        cls,
        base_url: str,
        timeout: float = 120.0,
        health_timeout: float = 3.0,
        default_task: str = "auto",
    ) -> "HeyiRemoteClient":
        """按 base_url 单例缓存，避免反复建 Session。"""
        key = f"{base_url}|{default_task}"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    base_url=base_url,
                    timeout=timeout,
                    health_timeout=health_timeout,
                    default_task=default_task,
                )
            return cls._instances[key]

    def is_alive(self) -> bool:
        """轻量健康检查：GET /health -> 期待 ``{"status":"ok",...}``。"""
        try:
            r = self._session.get(
                f"{self.base_url}/health", timeout=self.health_timeout
            )
        except Exception as e:  # noqa: BLE001
            logger.info("[HeyiRemote] /health 不可达: %s", e)
            return False

        if not r.ok:
            logger.warning(
                "[HeyiRemote] /health 非 2xx: status=%s body=%s",
                r.status_code,
                r.text[:200],
            )
            return False

        try:
            data = r.json()
        except ValueError:
            logger.warning("[HeyiRemote] /health 响应非 JSON: %s", r.text[:200])
            return False

        if data.get("status") != "ok":
            logger.warning("[HeyiRemote] /health 返回 status != ok: %s", data)
            return False
        return True

    def segment(
        self, image_path: str, task: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """上传图像 -> 下载 NIfTI mask -> 解析成 2D 二值 mask。

        Args:
            image_path: 本地 ``.png`` / ``.jpg`` / ``.nii`` / ``.nii.gz`` 文件
            task: ``"auto"`` / ``"hemorrhage"`` / ``"ischemia"``；
                  ``None`` 时使用 ``self.default_task``。

        Returns:
            (mask, orig_rgb):
              - mask:     ``np.uint8`` ``[H, W]`` 取值 0/1
              - orig_rgb: ``np.uint8`` ``[H, W, 3]`` 原图 RGB
              （形状已对齐，可直接喂给本地 ``HeyiVisionAdapter`` 的
              ``overlay`` / ``mask_statistics``）

        Raises:
            HeyiRemoteError: 任何一步失败时抛出，调用方应降级到本地。
        """
        p = Path(image_path)
        if not p.is_file():
            raise HeyiRemoteError(f"输入文件不存在: {image_path}")

        task_arg = task or self.default_task

        try:
            with p.open("rb") as f:
                files = {"file": (p.name, f, "application/octet-stream")}
                data = {"task": task_arg}
                resp = self._session.post(
                    f"{self.base_url}/segment",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
        except self._requests.exceptions.RequestException as e:
            raise HeyiRemoteError(f"/segment 网络异常: {e}") from e

        if not resp.ok:
            raise HeyiRemoteError(
                f"/segment HTTP {resp.status_code}: {resp.text[:300]}"
            )

        mask_bytes = resp.content
        if not mask_bytes:
            raise HeyiRemoteError("/segment 返回空响应体")

        try:
            orig_rgb = _load_input_as_rgb(image_path)
        except Exception as e:  # noqa: BLE001
            raise HeyiRemoteError(f"原图加载失败: {e}") from e

        try:
            mask = _parse_mask_nii_bytes(mask_bytes, target_hw=orig_rgb.shape[:2])
        except Exception as e:  # noqa: BLE001
            raise HeyiRemoteError(f"远程 NIfTI mask 解析失败: {e}") from e

        logger.info(
            "[HeyiRemote] segment 完成: task=%s mask_shape=%s fg_px=%d",
            task_arg,
            mask.shape,
            int(mask.sum()),
        )
        return mask, orig_rgb

    def segment_preview(
        self, image_path: str, task: Optional[str] = None
    ) -> bytes:
        """调用 ``/segment/preview`` 拿到服务端烤好的可视化图像字节流。

        与 :meth:`segment` 解耦：

        * ``/segment``         返回 NIfTI mask  -> 用于统计 / 诊断文本
        * ``/segment/preview`` 返回 PNG/JPEG 可视化 -> 用于前端展示的 "分割结果"

        Args:
            image_path: 本地图像路径
            task: ``"auto"`` / ``"hemorrhage"`` / ``"ischemia"``；
                  ``None`` 时使用 ``self.default_task``。

        Returns:
            图像二进制字节流（PNG 或 JPEG），调用方可直接写盘
            (``brain_stroke_plot.png``)。

        Raises:
            HeyiRemoteError: 任何一步失败时抛出。

        容错策略：
            1. 优先使用传入 task；如果服务端明确返回 4xx，自动重试一次
               不带 task 的请求（兼容服务端只在 ``/segment`` 接受 task 的实现）。
            2. PNG / JPEG 都视为合法图像；非图像内容才抛错。
        """
        p = Path(image_path)
        if not p.is_file():
            raise HeyiRemoteError(f"输入文件不存在: {image_path}")

        task_arg = task or self.default_task

        resp = self._post_segment_preview(p, task_arg)

        # 4xx：可能服务端不接受 task 字段，去掉重试一次
        if resp is not None and 400 <= resp.status_code < 500 and task_arg:
            logger.info(
                "[HeyiRemote] /segment/preview 拒绝 task=%s (HTTP %s)，"
                "去掉 task 重试一次以兼容老版本服务端。",
                task_arg,
                resp.status_code,
            )
            resp = self._post_segment_preview(p, None)

        if resp is None:
            raise HeyiRemoteError("/segment/preview 网络异常 (见前序日志)")

        if not resp.ok:
            raise HeyiRemoteError(
                f"/segment/preview HTTP {resp.status_code}: {resp.text[:300]}"
            )

        body = resp.content
        if not body:
            raise HeyiRemoteError("/segment/preview 返回空响应体")

        kind = _sniff_image_kind(body, resp.headers.get("Content-Type", ""))
        if kind is None:
            ctype = resp.headers.get("Content-Type", "")
            logger.warning(
                "[HeyiRemote] /segment/preview 返回非图像内容 (Content-Type=%s, head=%r)",
                ctype,
                body[:16],
            )
            raise HeyiRemoteError(
                f"/segment/preview 返回非图像内容 (Content-Type={ctype})"
            )

        logger.info(
            "[HeyiRemote] segment/preview 完成: task=%s kind=%s bytes=%d",
            task_arg,
            kind,
            len(body),
        )
        return body

    def _post_segment_preview(self, p: Path, task: Optional[str]):
        """调用 ``/segment/preview``；网络异常返回 None 而不抛。"""
        try:
            with p.open("rb") as f:
                files = {"file": (p.name, f, "application/octet-stream")}
                data: Dict[str, str] = {"task": task} if task else {}
                return self._session.post(
                    f"{self.base_url}/segment/preview",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
        except self._requests.exceptions.RequestException as e:
            logger.warning(
                "[HeyiRemote] /segment/preview 网络异常 (task=%s): %s", task, e
            )
            return None


def _load_input_as_rgb(image_path: str) -> np.ndarray:
    """把输入图像统一加载成 ``[H, W, 3]`` ``np.uint8`` RGB。"""
    lower = image_path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        import nibabel as nib  # noqa: WPS433

        nii = nib.load(image_path)
        arr = np.asarray(nii.get_fdata())
        return _nii_volume_to_rgb(arr)

    from PIL import Image  # noqa: WPS433

    pil = Image.open(image_path).convert("RGB")
    return np.array(pil)


def _nii_volume_to_rgb(arr: np.ndarray) -> np.ndarray:
    """NIfTI 体数据 -> 2D 灰度图（复制 3 通道） ``[H, W, 3]`` uint8。"""
    a = arr
    while a.ndim > 2:
        mid = a.shape[-1] // 2
        a = np.take(a, mid, axis=-1)

    a = a.astype(np.float32)
    a_min, a_max = float(a.min()), float(a.max())
    if a_max - a_min < 1e-6:
        gray = np.zeros_like(a, dtype=np.uint8)
    else:
        gray = ((a - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _parse_mask_nii_bytes(
    mask_bytes: bytes, target_hw: Tuple[int, int]
) -> np.ndarray:
    """把 /segment 返回的 ``.nii.gz`` 字节流解析成 ``[H,W]`` uint8 {0,1}。

    走临时文件路径是因为 nibabel 从 ``BytesIO`` 加载 gzipped NIfTI 在不同
    版本里行为不一致，而写一次小文件既可靠又没有明显成本。
    """
    import nibabel as nib  # noqa: WPS433

    suffix = ".nii.gz" if _looks_gzipped(mask_bytes) else ".nii"
    tmp_path: Optional[str] = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(mask_bytes)
            tmp_path = f.name

        nii = nib.load(tmp_path)
        arr = np.asarray(nii.get_fdata())
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    mask2d = _collapse_mask_to_2d(arr)
    mask2d = (mask2d > 0).astype(np.uint8)

    h, w = target_hw
    if mask2d.shape != (h, w):
        mask2d = _resize_mask_nn(mask2d, (h, w))

    return mask2d


def _collapse_mask_to_2d(arr: np.ndarray) -> np.ndarray:
    """Mask 可能是 2D/3D/4D，逐轴选"前景最多"的切片退化成 2D。"""
    a = arr
    while a.ndim > 2:
        axis = a.ndim - 1
        sums = (a > 0).reshape(-1, a.shape[axis]).sum(axis=0)
        idx = int(sums.argmax()) if sums.size > 0 and sums.max() > 0 else a.shape[axis] // 2
        a = np.take(a, idx, axis=axis)
    return a


def _resize_mask_nn(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """最近邻缩放，避免插值把二值 mask 搞成灰度。"""
    try:
        import cv2  # noqa: WPS433

        h, w = target_hw
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
    except Exception:  # noqa: BLE001
        from PIL import Image  # noqa: WPS433

        h, w = target_hw
        pil = Image.fromarray((mask * 255).astype(np.uint8))
        pil = pil.resize((w, h), resample=Image.NEAREST)
        return (np.array(pil) > 0).astype(np.uint8)


def _looks_gzipped(data: bytes) -> bool:
    """简单地按 gzip magic header 判断是否 gzip 压缩过。"""
    return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def _sniff_image_kind(body: bytes, content_type: str = "") -> Optional[str]:
    """识别 ``/segment/preview`` 返回的图像类型；都不像就返回 None。

    优先看 magic bytes（最可靠），其次看 ``Content-Type``。
    """
    if len(body) >= 8 and body[:8] == _PNG_MAGIC:
        return "png"
    if len(body) >= 3 and body[:3] == _JPEG_MAGIC:
        return "jpeg"
    ctype = (content_type or "").lower()
    if "png" in ctype:
        return "png"
    if "jpeg" in ctype or "jpg" in ctype:
        return "jpeg"
    return None


__all__ = ["HeyiRemoteClient", "HeyiRemoteError"]
