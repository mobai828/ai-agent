from .vit_encoder import ViTEncoder
from .resnet_encoder import ResNetEncoder
from architect.core.registry import ComponentRegistry

# 注册 ViT 编码器
ComponentRegistry.register_backbone('vit_encoder')(ViTEncoder)

# 注册 ResNet 编码器
ComponentRegistry.register_backbone('resnet_encoder')(ResNetEncoder)

# 导出编码器类
__all__ = ['ViTEncoder', 'ResNetEncoder']
