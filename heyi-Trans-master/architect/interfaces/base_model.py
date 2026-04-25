"""
基础模型接口 - 所有模型必须实现这些接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

class BaseModel(ABC):
    """模型基类
    
    所有模型的抽象基类，定义了模型的核心接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self.current_task = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, task_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """前向传播
        
        Args:
            x: 输入张量
            task_name: 任务名称（可选）
            **kwargs: 额外参数
            
        Returns:
            包含模型输出的字典
        """
        pass
    
    @abstractmethod
    def set_task(self, task_name: str) -> None:
        """设置当前任务
        
        Args:
            task_name: 任务名称
        """
        pass
    
    @abstractmethod
    def get_supported_tasks(self) -> list:
        """获取支持的任务列表
        
        Returns:
            支持的任务名称列表
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.config
    
    def to(self, device: torch.device):
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            移动后的模型
        """
        # 子类应实现具体的设备移动逻辑
        return self
