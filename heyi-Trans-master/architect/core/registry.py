"""
组件注册器 - 所有组件在这里注册
"""
from typing import Dict, Any, Callable, List, Optional

class ComponentRegistry:
    """组件注册中心
    
    管理所有组件的注册和获取，包括：
    - 骨干网络（backbones）
    - 任务头（heads）
    - 模型（models）
    """
    
    _backbones: Dict[str, Callable] = {}
    _heads: Dict[str, Callable] = {}
    _models: Dict[str, Callable] = {}
    
    @classmethod
    def register_backbone(cls, name: str):
        """注册骨干网络
        
        Args:
            name: 骨干网络名称
            
        Returns:
            装饰器函数
        """
        def decorator(builder: Callable[[Dict[str, Any]], Any]):
            """装饰器函数
            
            Args:
                builder: 构建骨干网络的函数
                
            Returns:
                原始构建函数
            """
            if not callable(builder):
                raise TypeError(f"骨干网络构建器必须是可调用对象，得到: {type(builder)}")
            
            if name in cls._backbones:
                raise ValueError(f"骨干网络名称 '{name}' 已被注册")
            
            cls._backbones[name] = builder
            return builder
        return decorator
    
    @classmethod
    def register_head(cls, task: str):
        """注册任务头
        
        Args:
            task: 任务名称
            
        Returns:
            装饰器函数
        """
        def decorator(builder: Callable[[Dict[str, Any]], Any]):
            """装饰器函数
            
            Args:
                builder: 构建任务头的函数
                
            Returns:
                原始构建函数
            """
            if not callable(builder):
                raise TypeError(f"任务头构建器必须是可调用对象，得到: {type(builder)}")
            
            if task in cls._heads:
                raise ValueError(f"任务头 '{task}' 已被注册")
            
            cls._heads[task] = builder
            return builder
        return decorator
    
    @classmethod
    def register_model(cls, name: str):
        """注册模型
        
        Args:
            name: 模型名称
            
        Returns:
            装饰器函数
        """
        def decorator(builder: Callable[[Dict[str, Any]], Any]):
            """装饰器函数
            
            Args:
                builder: 构建模型的函数
                
            Returns:
                原始构建函数
            """
            if not callable(builder):
                raise TypeError(f"模型构建器必须是可调用对象，得到: {type(builder)}")
            
            if name in cls._models:
                raise ValueError(f"模型名称 '{name}' 已被注册")
            
            cls._models[name] = builder
            return builder
        return decorator
    
    @classmethod
    def get_backbone(cls, name: str, config: Dict[str, Any]):
        """获取骨干网络
        
        Args:
            name: 骨干网络名称
            config: 配置字典
            
        Returns:
            实例化的骨干网络
            
        Raises:
            ValueError: 如果骨干网络不存在
        """
        if name not in cls._backbones:
            raise ValueError(f"未找到骨干网络: '{name}'。已注册的骨干网络: {list(cls._backbones.keys())}")
        
        try:
            return cls._backbones[name](config)
        except Exception as e:
            raise RuntimeError(f"创建骨干网络 '{name}' 时出错: {str(e)}") from e
    
    @classmethod
    def get_head(cls, task: str, config: Dict[str, Any]):
        """获取任务头
        
        Args:
            task: 任务名称
            config: 配置字典
            
        Returns:
            实例化的任务头
            
        Raises:
            ValueError: 如果任务头不存在
        """
        if task not in cls._heads:
            raise ValueError(f"未找到任务头: '{task}'。已注册的任务头: {list(cls._heads.keys())}")
        
        try:
            return cls._heads[task](config)
        except Exception as e:
            raise RuntimeError(f"创建任务头 '{task}' 时出错: {str(e)}") from e
    
    @classmethod
    def get_model(cls, name: str, config: Dict[str, Any]):
        """获取模型
        
        Args:
            name: 模型名称
            config: 配置字典
            
        Returns:
            实例化的模型
            
        Raises:
            ValueError: 如果模型不存在
        """
        if name not in cls._models:
            raise ValueError(f"未找到模型: '{name}'。已注册的模型: {list(cls._models.keys())}")
        
        try:
            return cls._models[name](config)
        except Exception as e:
            raise RuntimeError(f"创建模型 '{name}' 时出错: {str(e)}") from e
    
    @classmethod
    def get_registered_backbones(cls) -> List[str]:
        """获取所有已注册的骨干网络名称
        
        Returns:
            骨干网络名称列表
        """
        return list(cls._backbones.keys())
    
    @classmethod
    def get_registered_heads(cls) -> List[str]:
        """获取所有已注册的任务头名称
        
        Returns:
            任务头名称列表
        """
        return list(cls._heads.keys())
    
    @classmethod
    def get_registered_models(cls) -> List[str]:
        """获取所有已注册的模型名称
        
        Returns:
            模型名称列表
        """
        return list(cls._models.keys())
    
    @classmethod
    def clear_registry(cls):
        """清空注册中心（主要用于测试）"""
        cls._backbones.clear()
        cls._heads.clear()
        cls._models.clear()
