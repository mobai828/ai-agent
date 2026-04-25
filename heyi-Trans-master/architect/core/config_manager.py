"""
配置管理器 - 加载和验证配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理
    
    负责加载、验证和管理模型配置
    支持配置继承和合并
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            加载并合并后的配置字典
            
        Raises:
            FileNotFoundError: 如果配置文件不存在
            yaml.YAMLError: 如果配置文件格式错误
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError(f"配置文件为空: {config_path}")
            
            # 处理继承
            if '_base_' in config:
                base_path = path.parent / config['_base_']
                base_config = ConfigManager.load_config(str(base_path))
                # 合并配置
                config = ConfigManager.merge_configs(base_config, config)
                del config['_base_']
            
            # 验证配置
            ConfigManager.validate_config(config)
            
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {config_path}, 错误: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {config_path}, 错误: {str(e)}") from e
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """验证配置
        
        Args:
            config: 配置字典
            
        Raises:
            ValueError: 如果配置不符合要求
        """
        # 验证必要的配置项
        if 'model' not in config:
            raise ValueError("配置中缺少 'model' 部分")
        
        model_config = config['model']
        if 'vision_encoder' not in model_config:
            raise ValueError("配置中缺少 'vision_encoder' 部分")
        
        if 'tasks' not in model_config:
            raise ValueError("配置中缺少 'tasks' 部分")
        
        # 验证任务配置
        tasks_config = model_config['tasks']
        if not isinstance(tasks_config, dict):
            raise ValueError("'tasks' 必须是字典类型")
        
        # 验证编码器配置
        encoder_config = model_config['vision_encoder']
        if 'type' not in encoder_config:
            raise ValueError("编码器配置中缺少 'type' 字段")
    
    @staticmethod
    def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """获取模型配置
        
        Args:
            config: 完整配置
            
        Returns:
            模型配置
        """
        return config.get('model', {})
    
    @staticmethod
    def get_encoder_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """获取编码器配置
        
        Args:
            config: 完整配置
            
        Returns:
            编码器配置
        """
        model_config = ConfigManager.get_model_config(config)
        return model_config.get('vision_encoder', {})
    
    @staticmethod
    def get_task_config(config: Dict[str, Any], task_name: str) -> Optional[Dict[str, Any]]:
        """获取特定任务的配置
        
        Args:
            config: 完整配置
            task_name: 任务名称
            
        Returns:
            任务配置，如果不存在则返回 None
        """
        model_config = ConfigManager.get_model_config(config)
        tasks_config = model_config.get('tasks', {})
        return tasks_config.get(task_name)
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """保存配置到文件
        
        Args:
            config: 配置字典
            save_path: 保存路径
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
