"""
统一模型 - 整合所有组件
"""
import torch.nn as nn
from typing import Dict, Any, Optional
import torch
from .interfaces.base_model import BaseModel
from .interfaces.base_encoder import BaseEncoder
from .interfaces.base_head import BaseHead

class UniversalVisionModel(BaseModel, nn.Module):
    """
    通用视觉模型
    
    设计原则：
    1. 配置驱动
    2. 模块化
    3. 支持多任务
    
    支持的任务：
    - 分类
    - 检测
    - 分割
    - 检索
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化通用视觉模型
        
        Args:
            config: 模型配置字典
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)
        
        # 初始化组件
        self._init_components(config)
        
    def _init_components(self, config: Dict[str, Any]):
        """初始化所有组件
        
        Args:
            config: 模型配置字典
        """
        try:
            # 1. 视觉编码器
            from .core.registry import ComponentRegistry
            encoder_type = config['model']['vision_encoder']['type']
            encoder_config = config['model']['vision_encoder']
            self.vision_encoder = ComponentRegistry.get_backbone(
                encoder_type,
                encoder_config
            )
            
            # 2. 任务头
            self.task_heads = nn.ModuleDict()
            for task_name, task_config in config['model']['tasks'].items():
                if task_config.get('enabled', False):
                    self.task_heads[task_name] = ComponentRegistry.get_head(
                        task_name,
                        task_config
                    )
        except Exception as e:
            raise RuntimeError(f"初始化模型组件失败: {str(e)}") from e
    
    def set_task(self, task_name: str) -> None:
        """设置当前任务
        
        Args:
            task_name: 任务名称
            
        Raises:
            ValueError: 如果任务不存在
        """
        if task_name not in self.task_heads:
            raise ValueError(f"未知任务: {task_name}。支持的任务: {list(self.task_heads.keys())}")
        self.current_task = task_name
    
    def forward(self, x: torch.Tensor, task_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """前向传播
        
        Args:
            x: 输入张量
            task_name: 任务名称（可选）
            **kwargs: 额外参数
            
        Returns:
            包含模型输出的字典
            
        Raises:
            ValueError: 如果未指定任务
        """
        # 确定任务
        if task_name is None:
            task_name = self.current_task
        if task_name is None:
            raise ValueError("必须指定任务名称")
        
        if task_name not in self.task_heads:
            raise ValueError(f"未知任务: {task_name}。支持的任务: {list(self.task_heads.keys())}")
        
        # 提取特征
        features = self.vision_encoder(x)
        
        # 任务特定处理
        outputs = self.task_heads[task_name](features, **kwargs)
        outputs['task'] = task_name
        
        return outputs
    
    def get_supported_tasks(self) -> list:
        """获取支持的任务列表
        
        Returns:
            支持的任务名称列表
        """
        return list(self.task_heads.keys())
    
    def to(self, device: torch.device):
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            移动后的模型
        """
        # 移动编码器
        if hasattr(self, 'vision_encoder'):
            self.vision_encoder = self.vision_encoder.to(device)
        
        # 移动任务头
        if hasattr(self, 'task_heads'):
            for task_name, head in self.task_heads.items():
                self.task_heads[task_name] = head.to(device)
        
        return super().to(device)
    
    def get_encoder_output_dim(self) -> int:
        """获取编码器输出维度
        
        Returns:
            编码器输出维度
        """
        if hasattr(self, 'vision_encoder'):
            return self.vision_encoder.get_output_dim()
        raise RuntimeError("编码器未初始化")
    
    def get_task_head(self, task_name: str) -> BaseHead:
        """获取指定任务的头
        
        Args:
            task_name: 任务名称
            
        Returns:
            任务头实例
            
        Raises:
            ValueError: 如果任务不存在
        """
        if task_name not in self.task_heads:
            raise ValueError(f"未知任务: {task_name}。支持的任务: {list(self.task_heads.keys())}")
        return self.task_heads[task_name]
