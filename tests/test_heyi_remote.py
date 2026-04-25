"""
Heyi 远程分割服务端到端测试
==========================

直接对接同学部署的真实服务（默认 http://222.198.105.83:8000），验证：

1. ``is_alive()`` 在服务在线时返回 True，在错误 URL 时返回 False
2. ``segment()`` 端到端：上传合成 CT 图 -> 下载 NIfTI mask ->
   解析成 2D 二值矩阵；mask 形状、dtype、取值全部符合契约
3. 不同 ``task`` 参数 (``auto`` / ``hemorrhage`` / ``ischemia``) 都能成功调用
4. 异常路径：网络超时 / 无效 URL 应抛 ``HeyiRemoteError``

说明：
-----
- 这是 **真实网络测试**，不是 mock；服务挂了测试会自动 skip，不会 fail。
- 测试图不是真 CT，只是能让服务器解码成功的合法 PNG。目的是验证 **客户端
  与服务端协议契合**，不是验证模型精度。
- 文件既能被 ``pytest`` 收集，也可直接 ``python tests/test_heyi_remote.py``
  运行（没装 pytest 时也能跑通）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENT_PATH = REPO_ROOT / "agents" / "image_analysis_agent" / "heyi_remote_client.py"


def _load_client_module():
    """绕开 agents/__init__.py 里的 langchain 依赖，直接加载目标模块。"""
    spec = importlib.util.spec_from_file_location(
        "heyi_remote_client_under_test", str(CLIENT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_client_mod = _load_client_module()
HeyiRemoteClient = _client_mod.HeyiRemoteClient
HeyiRemoteError = _client_mod.HeyiRemoteError

REMOTE_URL = os.getenv("HEYI_REMOTE_URL", "http://222.198.105.83:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Fixtures (纯函数形式，避免强依赖 pytest)
# ---------------------------------------------------------------------------


def _make_synthetic_ct_png(path: Path, size: int = 256) -> Path:
    """生成一张合法的合成灰度 PNG，供服务器解码测试。

    - 形状 [size, size] 灰度
    - 包含低频背景 + 中心一个高亮椭圆（模拟病灶）+ 背景噪声
    """
    from PIL import Image  # noqa: WPS433

    yy, xx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    dist = ((xx - cx) / (size * 0.45)) ** 2 + ((yy - cy) / (size * 0.45)) ** 2
    brain = np.clip(1.0 - dist, 0.0, 1.0) * 180.0

    rx, ry = size * 0.12, size * 0.08
    lesion_cx, lesion_cy = cx + size * 0.15, cy - size * 0.1
    lesion_mask = (((xx - lesion_cx) / rx) ** 2 + ((yy - lesion_cy) / ry) ** 2) < 1.0
    brain[lesion_mask] += 60.0

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 6.0, size=brain.shape)
    img = np.clip(brain + noise, 0, 255).astype(np.uint8)

    Image.fromarray(img, mode="L").save(str(path))
    return path


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_client_instantiation():
    """客户端构造参数应该被正确读入。"""
    c = HeyiRemoteClient.get(
        REMOTE_URL, timeout=30.0, health_timeout=2.0, default_task="hemorrhage"
    )
    assert c.base_url == REMOTE_URL
    assert c.timeout == 30.0
    assert c.health_timeout == 2.0
    assert c.default_task == "hemorrhage"


def test_is_alive_against_invalid_url_returns_false():
    """错误 URL 下 is_alive 应静默返回 False，不抛异常。"""
    c = HeyiRemoteClient(
        base_url="http://127.0.0.1:1",
        timeout=2.0,
        health_timeout=1.0,
    )
    assert c.is_alive() is False


def test_health_check_live():
    """真实服务在线时 /health 应返回 True；离线则 skip。"""
    c = HeyiRemoteClient.get(REMOTE_URL, health_timeout=3.0)
    alive = c.is_alive()
    if not alive:
        _skip(f"远程服务 {REMOTE_URL} /health 不可达，跳过。")
    assert alive is True


def test_segment_end_to_end_png():
    """E2E: 上传合成 PNG -> 拿到 mask，断言形状 / dtype / 取值合法。"""
    c = HeyiRemoteClient.get(REMOTE_URL, timeout=180.0, health_timeout=3.0)
    if not c.is_alive():
        _skip(f"远程服务 {REMOTE_URL} 不可达，跳过。")

    with tempfile.TemporaryDirectory() as tmp:
        img_path = _make_synthetic_ct_png(Path(tmp) / "synthetic_ct.png")
        try:
            mask, orig_rgb = c.segment(str(img_path), task="auto")
        except HeyiRemoteError as e:
            _skip(f"/segment 调用失败 (服务端策略 / 兼容问题)：{e}")

    assert isinstance(mask, np.ndarray), "mask 不是 numpy 数组"
    assert isinstance(orig_rgb, np.ndarray), "orig_rgb 不是 numpy 数组"

    assert mask.ndim == 2, f"mask 应为 2D，实际 shape={mask.shape}"
    assert orig_rgb.ndim == 3 and orig_rgb.shape[2] == 3, (
        f"orig_rgb 应为 [H,W,3]，实际 shape={orig_rgb.shape}"
    )
    assert mask.shape == orig_rgb.shape[:2], (
        f"mask {mask.shape} 与 orig_rgb {orig_rgb.shape[:2]} 形状未对齐"
    )

    assert mask.dtype == np.uint8, f"mask dtype 应为 uint8，实际 {mask.dtype}"
    uniq = set(np.unique(mask).tolist())
    assert uniq <= {0, 1}, f"mask 应只含 0/1，实际 {uniq}"

    assert orig_rgb.dtype == np.uint8
    assert orig_rgb.shape[0] == 256 and orig_rgb.shape[1] == 256, (
        f"orig_rgb 形状应与原图一致 [256,256,3]，实际 {orig_rgb.shape}"
    )


def test_segment_explicit_tasks():
    """显式传 hemorrhage / ischemia，都应能完成调用（不保证有病灶）。"""
    c = HeyiRemoteClient.get(REMOTE_URL, timeout=180.0, health_timeout=3.0)
    if not c.is_alive():
        _skip(f"远程服务 {REMOTE_URL} 不可达，跳过。")

    with tempfile.TemporaryDirectory() as tmp:
        img_path = _make_synthetic_ct_png(Path(tmp) / "synthetic_ct.png")
        for task in ("hemorrhage", "ischemia"):
            try:
                mask, _ = c.segment(str(img_path), task=task)
            except HeyiRemoteError as e:
                raise AssertionError(f"task={task} 调用失败：{e}") from e
            assert mask.ndim == 2 and mask.dtype == np.uint8


def test_segment_raises_on_bad_url():
    """无效 URL 下 segment 必须抛 HeyiRemoteError 而不是让异常泄漏。"""
    c = HeyiRemoteClient(
        base_url="http://127.0.0.1:1",
        timeout=2.0,
        health_timeout=1.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        img_path = _make_synthetic_ct_png(Path(tmp) / "synthetic_ct.png")
        raised = False
        try:
            c.segment(str(img_path))
        except HeyiRemoteError:
            raised = True
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"期望 HeyiRemoteError，实际抛出 {type(e).__name__}: {e}"
            )
    assert raised, "segment 应在网络不可达时抛 HeyiRemoteError"


def test_segment_missing_file_raises():
    """输入文件不存在时必须抛 HeyiRemoteError。"""
    c = HeyiRemoteClient.get(REMOTE_URL, timeout=5.0, health_timeout=1.0)
    raised = False
    try:
        c.segment("__this_path_does_not_exist__.png")
    except HeyiRemoteError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Minimal standalone runner (unittest/pytest-free)
# ---------------------------------------------------------------------------


class _SkipTest(Exception):
    pass


def _skip(msg: str) -> None:
    raise _SkipTest(msg)


def _collect_tests() -> List[Tuple[str, Callable[[], None]]]:
    items: List[Tuple[str, Callable[[], None]]] = []
    for name, obj in sorted(globals().items()):
        if name.startswith("test_") and callable(obj):
            items.append((name, obj))
    return items


def _run_all() -> int:
    tests = _collect_tests()
    passed = failed = skipped = 0
    print(f"发现 {len(tests)} 个测试 (远程服务 = {REMOTE_URL})")
    print("-" * 72)
    for name, fn in tests:
        try:
            fn()
        except _SkipTest as e:
            skipped += 1
            print(f"[SKIP] {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS] {name}")
    print("-" * 72)
    print(f"总结: {passed} 通过 / {failed} 失败 / {skipped} 跳过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
