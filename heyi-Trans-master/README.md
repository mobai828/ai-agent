# Universal Vision Transformer (`heyi-Trans-master`)

> 一个基于 PyTorch 的**通用视觉大模型框架**，通过同一套 Backbone 支持
> **分类 / 检测 / 分割 / 检索** 四类任务，采用**配置驱动 + 模块化 + 组件注册**的设计。
>
> 本项目在上层仓库 [`agentgithub` (Multi-Agent Medical Assistant)](../README.md) 中
> 作为 **脑肿瘤 / 脑卒中分割模型的算法底座** 被接入，具体接入说明见
> [主仓库 README 的 Medical Imaging Models 章节](../README.md#-medical-imaging-models-heyi-trans-master-integration)。

---

## 📦 项目结构

```
heyi-Trans-master/
├── architect/                  # 核心架构层
│   ├── core/
│   │   ├── config_manager.py   # YAML 配置加载 + 继承合并 + 校验
│   │   └── registry.py         # ComponentRegistry，通过装饰器注册 backbone / head / model
│   ├── interfaces/             # 抽象基类：BaseModel / BaseEncoder / BaseHead
│   └── unified_model.py        # UniversalVisionModel：按配置组装 encoder + task heads
├── backbone/                   # 视觉编码器（主干网络）
│   ├── encoders/
│   │   ├── vit_encoder.py      # ViT-B/16, ViT-L/16, ViT-B/32
│   │   └── resnet_encoder.py   # ResNet50 / ResNet101
│   └── fusion/
│       └── feature_adapter.py  # 把不同 backbone 输出统一为 FPN 格式
├── heads/                      # 四个任务头
│   ├── classification/
│   ├── detection/
│   ├── segmentation/
│   └── retrieval/
├── training/                   # 训练流水线
│   ├── data/                   # MultiTaskDataset + transforms
│   └── scripts/
│       ├── train.py            # 多任务训练入口
│       └── evaluate.py
├── api/                        # FastAPI 推理服务（独立于主仓库的 app.py）
│   ├── app.py
│   ├── model_handler.py
│   └── preprocessing.py
├── evaluation/                 # 离线评估脚本
├── export/                     # 模型导出
├── configs/                    # YAML 配置
│   ├── base.yaml
│   ├── training_config.yaml
│   └── h20_config.yaml
├── examples/
│   └── minimal_example.py
├── tests/
└── requirements.txt
```

## 🧠 核心设计

### 1. 组件注册中心

所有 backbone / head / model 通过装饰器注册：

```python
from architect.core.registry import ComponentRegistry

@ComponentRegistry.register_head("segmentation")
class SegmentationHead(BaseHead, nn.Module):
    ...

# 业务侧按字符串取组件
encoder = ComponentRegistry.get_backbone("vit_encoder", encoder_cfg)
head    = ComponentRegistry.get_head("segmentation", head_cfg)
```

### 2. 统一模型

`UniversalVisionModel` 按配置文件组装 *一个编码器 + N 个任务头*：

```python
model = UniversalVisionModel(config)
model.set_task("classification")
outputs = model(images)  # {'logits': ..., 'task': 'classification'}
```

### 3. 配置示例

```yaml
# configs/base.yaml
model:
  vision_encoder:
    type: vit_encoder
    model_name: vit_b_16
    pretrained: true
  tasks:
    classification:
      enabled: true
      num_classes: 1000
      input_dim: 768
    segmentation:
      enabled: true
      num_classes: 21
      input_dim: 768
```

## 🚀 独立使用

### 最小运行示例

```bash
cd heyi-Trans-master
python examples/minimal_example.py
```

### 训练

```bash
cd heyi-Trans-master
python training/scripts/train.py --config configs/training_config.yaml
```

⚠️ **注意**：`training/scripts/train.py` 依赖一个能读多任务数据的 `MultiTaskDataset`，
当前实现尚未覆盖全部真实数据集（BraTS / COCO / ImageNet 等）的加载细节，需要根据你
自己的数据做 **数据加载适配**。

### 推理服务（独立）

```bash
cd heyi-Trans-master
python api/app.py
```

提供 `/health`、`/models`、`/inference`、`/batch_inference` 等接口。此服务与上层
`agentgithub` 主服务（端口 8000）相互独立，**不冲突**。

## 🧩 被主项目 `agentgithub` 引用的方式

主项目不直接调 `UniversalVisionModel`，而是通过一个轻量**桥接层**复用部分组件：

```python
# agents/image_analysis_agent/heyi_adapter.py （位于主仓库）

_HEYI_ROOT = "<agentgithub>/heyi-Trans-master"
sys.path.insert(0, _HEYI_ROOT)

import backbone.encoders                           # 触发 ComponentRegistry 注册
from architect.core.registry import ComponentRegistry

encoder = ComponentRegistry.get_backbone("vit_encoder", {
    "model_name": "vit_b_16",
    "pretrained": True,
})
# 主项目自带一个轻量二值分割解码器，直接接在 encoder 输出的 patch_features 上
```

**为什么不直接用 `UniversalVisionModel`？**

1. `heads/segmentation/segmentation_head.py` 使用三级相对导入
   `from ...architect...`，要求 `heyi-Trans-master` 目录自身是一个 Python 包，
   但当前仓库中这一层**没有顶层 `__init__.py`**。
2. heyi 自带的 `SegmentationHead` 是 FCN / `Conv2d` 结构，而 `ViTEncoder`
   输出的是序列式 `patch_features (B, N, D)`，**形状不匹配**。

所以主项目选择只复用 heyi 的 `ComponentRegistry` + `ViTEncoder`，自行接一个
和 ViT patch features 形状匹配的轻量二值分割头，作为医疗影像分割的运行实体。

## 🔬 当前状态（已知限制）

| 模块 | 状态 |
|------|------|
| `architect/`（Registry / Config / 抽象基类） | ✅ 完整可用 |
| `backbone/encoders/`（ViT / ResNet） | ✅ 完整可用，被主项目生产引用 |
| `backbone/fusion/feature_adapter.py` | ✅ 可用 |
| `heads/*` | ⚠️ 存在三级相对导入，且与 ViT patch-feature 形状不匹配，需要重构才能独立跑通 |
| `training/scripts/train.py` | ⚠️ 数据加载器仍需针对真实数据集补齐 |
| `api/app.py` | ⚠️ 独立推理服务骨架，配合 `ModelHandler` 使用；主项目不依赖此服务 |
| 预训练权重 | ❌ 仓库**不包含**任何权重文件，需要自行训练或下载开源医学分割权重 |

## 📄 许可证

本子项目随上层 `agentgithub` 主仓库发布，遵循同一 Apache 2.0 License，
详见 [../LICENSE](../LICENSE)。
