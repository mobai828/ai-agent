"""
最小示例 - 展示如何使用架构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from architect.core.config_manager import ConfigManager
from architect.unified_model import UniversalVisionModel

def main():
    print("=== Vision Transformer 最小示例 ===")
    
    # 1. 加载配置
    # 获取当前文件所在目录的父目录（即 vision_thransformer 目录）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "base.yaml")
    config = ConfigManager.load_config(config_path)
    print(f"项目: {config['project']['name']}")
    print(f"版本: {config['project']['version']}")
    
    # 2. 创建模型（稍后需要注册组件）
    print("创建模型...")
    model = UniversalVisionModel(config)
    
    # 3. 查看支持的任务
    tasks = model.get_supported_tasks()
    print(f"支持的任务: {tasks}")
    
    # 4. 设置任务
    if 'classification' in tasks:
        model.set_task('classification')
        
        # 5. 模拟输入
        images = torch.randn(2, 3, 224, 224)
        print(f"输入形状: {images.shape}")
        
        # 6. 前向传播
        try:
            outputs = model(images)
            print(f"输出类型: {type(outputs)}")
            print(f"任务: {outputs.get('task', 'unknown')}")
        except Exception as e:
            print(f"前向传播错误（正常，因为组件还没实现）: {e}")
    
    print("✅ 架构示例运行成功！")

if __name__ == "__main__":
    main()
