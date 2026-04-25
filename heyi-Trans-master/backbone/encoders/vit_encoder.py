import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from architect.interfaces.base_encoder import BaseEncoder

class ViTEncoder(BaseEncoder, nn.Module):
    """Vision Transformer (ViT) 编码器
    
    支持 ViT-B/16, ViT-L/16 等。
    提取 [CLS] token 和 patch 特征。
    """
    
    def __init__(self, config: Dict[str, Any]):
        BaseEncoder.__init__(self, config)
        nn.Module.__init__(self)
        
        self.model_name = config.get('model_name', 'vit_b_16')
        self.pretrained = config.get('pretrained', True)
        
        # 加载 ViT 模型
        if self.model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.DEFAULT if self.pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
        elif self.model_name == 'vit_l_16':
            weights = models.ViT_L_16_Weights.DEFAULT if self.pretrained else None
            self.backbone = models.vit_l_16(weights=weights)
        elif self.model_name == 'vit_b_32':
             weights = models.ViT_B_32_Weights.DEFAULT if self.pretrained else None
             self.backbone = models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"不支持的 ViT 模型: {self.model_name}")
            
        self.hidden_dim = self.backbone.hidden_dim
        self.patch_size = self.backbone.patch_size
        
        # 如果需要，冻结骨干网络参数
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播以提取 ViT 特征"""
        # 使用 torchvision 的内部方法重塑和处理输入
        # 这会将输入图像投影为 patch embeddings 并展平
        x = self.backbone._process_input(x)
        n = x.shape[0]
        
        # 将 class token 扩展到整个 batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 通过编码器 (Transformer 层)
        x = self.backbone.encoder(x)
        
        # x 的形状是 [batch, seq_len, hidden_dim]
        # seq_len = 1 + (H/P * W/P)
        
        cls_token = x[:, 0]
        patch_features = x[:, 1:]
        
        return {
            "cls_token": cls_token,
            "patch_features": patch_features,
            # 同时返回完整序列以备不时之需
            "last_hidden_state": x
        }
    
    def get_output_dim(self) -> int:
        return self.hidden_dim
        
    def to(self, device: torch.device):
        return super(ViTEncoder, self).to(device)
