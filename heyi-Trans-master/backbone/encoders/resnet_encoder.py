import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from architect.interfaces.base_encoder import BaseEncoder

class ResNetEncoder(BaseEncoder, nn.Module):
    """ResNet 骨干网络编码器
    
    支持 ResNet50, ResNet101 等。
    提取多尺度特征 (C2, C3, C4, C5)。
    """
    
    def __init__(self, config: Dict[str, Any]):
        BaseEncoder.__init__(self, config)
        nn.Module.__init__(self)
        
        self.depth = config.get('depth', 50)
        self.pretrained = config.get('pretrained', True)
        
        # 加载 ResNet 模型
        if self.depth == 50:
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif self.depth == 101:
            weights = models.ResNet101_Weights.DEFAULT if self.pretrained else None
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的 ResNet 深度: {self.depth}")
            
        # 如果配置了冻结层，则冻结指定阶段
        frozen_stages = config.get('frozen_stages', -1)
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

        # 移除分类头 (fc) 和 avgpool，因为特征提取不需要它们
        # 但我们保留 self.backbone 中的模块以便在 forward 中使用
        del self.backbone.fc
        del self.backbone.avgpool

    def _freeze_stages(self, frozen_stages: int):
        """冻结网络的前几个阶段"""
        if frozen_stages >= 0:
            self.backbone.bn1.eval()
            for m in [self.backbone.conv1, self.backbone.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        
        for i in range(1, frozen_stages + 1):
            m = getattr(self.backbone, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播以提取多尺度特征"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)  # Stride 4 (下采样4倍)
        c3 = self.backbone.layer2(c2) # Stride 8 (下采样8倍)
        c4 = self.backbone.layer3(c3) # Stride 16 (下采样16倍)
        c5 = self.backbone.layer4(c4) # Stride 32 (下采样32倍)
        
        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5
        }
    
    def get_output_dim(self) -> int:
        """返回最后一个特征图 (C5) 的通道维度"""
        return self.backbone.layer4[-1].conv3.out_channels
    
    def get_feature_dims(self) -> Dict[str, int]:
        """返回所有特征图的通道维度"""
        return {
            "c2": 256,
            "c3": 512,
            "c4": 1024,
            "c5": 2048
        }
        
    def to(self, device: torch.device):
        return super(ResNetEncoder, self).to(device)
