"""
编码器接口 - 视觉/文本编码器
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

class BaseEncoder(ABC):
    """编码器基类
    
    所有编码器的抽象基类，定义了编码器的核心接口
    支持ResNet、ViT等不同类型的骨干网络
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化编码器
        
        Args:
            config: 编码器配置字典
        """
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取特征
        
        Args:
            x: 输入张量
            
        Returns:
            包含不同层级特征的字典
        """
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """获取输出维度
        
        Returns:
            编码器输出特征的维度
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取编码器配置
        
        Returns:
            编码器配置字典
        """
        return self.config
    
    def to(self, device: torch.device):
        """将编码器移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            移动后的编码器
        """
        # 子类应实现具体的设备移动逻辑
        return self
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取特征图
        
        Args:
            x: 输入张量
            
        Returns:
            包含不同层级特征图的字典
        """
        # 默认实现，直接调用forward
        return self.forward(x)
