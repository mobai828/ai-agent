"""
任务头接口 - 分类/检测/分割/检索头
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch

class BaseHead(ABC):
    """任务头基类
    
    所有任务头的抽象基类，定义了任务头的核心接口
    支持分类、检测、分割、检索4个任务
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化任务头
        
        Args:
            config: 任务头配置字典
        """
        self.config = config
    
    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor], 
                targets: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """前向传播
        
        Args:
            features: 编码器提取的特征字典
            targets: 目标标签（可选）
            **kwargs: 额外参数
            
        Returns:
            包含模型输出的字典
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """计算损失
        
        Args:
            predictions: 模型预测结果
            targets: 目标标签
            
        Returns:
            损失张量
        """
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        """获取任务类型
        
        Returns:
            任务类型名称
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取任务头配置
        
        Returns:
            任务头配置字典
        """
        return self.config
    
    def to(self, device: torch.device):
        """将任务头移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            移动后的任务头
        """
        # 子类应实现具体的设备移动逻辑
        return self
    
    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """后处理预测结果
        
        Args:
            predictions: 模型预测结果
            
        Returns:
            后处理后的结果
        """
        # 默认实现，直接返回原始预测
        return predictions
