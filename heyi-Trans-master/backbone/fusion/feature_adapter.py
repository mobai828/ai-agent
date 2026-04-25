import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union

class FeatureAdapter(nn.Module):
    """
    特征适配器 & FPN (特征金字塔网络) 模块
    
    将不同骨干网络 (ResNet, ViT) 的输出统一为标准的 FPN 格式。
    标准输出: {'c2': ..., 'c3': ..., 'c4': ..., 'c5': ...}
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.backbone_type = config.get('backbone_type', 'resnet')
        self.out_channels = config.get('out_channels', 256)
        
        in_channels = config.get('in_channels') # 可以是字典或整数
        
        if self.backbone_type == 'resnet':
            if not isinstance(in_channels, dict):
                raise ValueError("对于 ResNet，in_channels 必须是 {layer_name: channels} 的字典")
                
            self.fpn_laterals = nn.ModuleDict()
            self.fpn_outputs = nn.ModuleDict()
            
            # 我们假设输入键是 c2, c3, c4, c5
            for name in ['c2', 'c3', 'c4', 'c5']:
                if name in in_channels:
                    # 侧向连接：将不同维度的特征映射到统一的 out_channels
                    self.fpn_laterals[name] = nn.Conv2d(in_channels[name], self.out_channels, 1)
                    # 输出卷积：消除上采样带来的混叠效应
                    self.fpn_outputs[name] = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
                
        elif self.backbone_type == 'vit':
            if not isinstance(in_channels, int):
                # 如果传入的是字典，尝试获取 hidden_dim
                if isinstance(in_channels, dict) and 'hidden_dim' in in_channels:
                    self.in_dim = in_channels['hidden_dim']
                else:
                    raise ValueError("对于 ViT，in_channels 必须是整数 (hidden dimension) 或包含 'hidden_dim' 的字典")
            else:
                self.in_dim = in_channels
            
            # 针对 ViT 的简单 FPN 实现
            # ViT 输出通常是 1/16 stride (相当于 C4)
            
            # C4 (1/16) 投影
            self.scale_p4 = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_channels, 1),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
            
            # 上采样得到 C3 (1/8)
            self.up_p3 = nn.Sequential(
                nn.ConvTranspose2d(self.out_channels, self.out_channels, 2, stride=2),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
            
            # 上采样得到 C2 (1/4)
            self.up_p2 = nn.Sequential(
                nn.ConvTranspose2d(self.out_channels, self.out_channels, 2, stride=2),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
            
            # 下采样得到 C5 (1/32)
            self.down_p5 = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
            
        else:
            raise ValueError(f"未知的 backbone 类型: {self.backbone_type}")
            
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.backbone_type == 'resnet':
            return self._forward_resnet(inputs)
        elif self.backbone_type == 'vit':
            return self._forward_vit(inputs)
        else:
            return inputs

    def _forward_resnet(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 标准 FPN 自顶向下路径
        # 准备侧向连接特征
        laterals = {}
        for name in ['c2', 'c3', 'c4', 'c5']:
            if name in inputs and name in self.fpn_laterals:
                laterals[name] = self.fpn_laterals[name](inputs[name])
        
        # 自顶向下路径
        # C5 是最顶层
        used_levels = sorted(laterals.keys(), key=lambda x: int(x[1]), reverse=True) # ['c5', 'c4', 'c3', 'c2']
        
        if not used_levels:
            return {}

        prev_features = laterals[used_levels[0]]
        results = {used_levels[0]: self.fpn_outputs[used_levels[0]](prev_features)}
        
        for i in range(1, len(used_levels)):
            level = used_levels[i]
            lat = laterals[level]
            
            # 上采样上一层特征并与当前层相加
            top_down = F.interpolate(prev_features, size=lat.shape[-2:], mode='nearest')
            feat = lat + top_down
            
            results[level] = self.fpn_outputs[level](feat)
            prev_features = feat # 使用融合后的特征进行下一层计算
            
        return results

    def _forward_vit(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        patch_features = inputs['patch_features'] # [B, N, D]
        B, N, D = patch_features.shape
        
        # 重塑为 2D 特征图
        H_feat = int(N**0.5)
        W_feat = H_feat
        
        x = patch_features.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        
        # 基础特征 (P4 - 1/16)
        p4 = self.scale_p4(x)
        
        # P3
        p3 = self.up_p3(p4)
        
        # P2
        p2 = self.up_p2(p3)
        
        # P5 (从 P4 下采样)
        p5 = self.down_p5(p4)
        
        return {
            "c2": p2,
            "c3": p3,
            "c4": p4,
            "c5": p5
        }
        
    def get_feature_dims(self) -> Dict[str, int]:
        return {
            "c2": self.out_channels,
            "c3": self.out_channels,
            "c4": self.out_channels,
            "c5": self.out_channels
        }
