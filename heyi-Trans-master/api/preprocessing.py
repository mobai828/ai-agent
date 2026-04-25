#!/usr/bin/env python3
"""
图像预处理模块

功能：
- 加载和处理输入图像
- 执行图像预处理步骤
- 转换为模型输入格式
"""

import numpy as np
import io
from PIL import Image

# 图像预处理函数
def preprocess_image(image_data):
    """预处理输入图像
    
    Args:
        image_data: 图像数据（字节流）
    
    Returns:
        预处理后的输入张量
    
    Raises:
        ValueError: 如果图像处理失败
    """
    try:
        # 从字节流加载图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整图像大小
        image = image.resize((224, 224))
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 归一化
        image_array = image_array / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # 调整维度顺序 (H, W, C) -> (C, H, W)
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # 添加批次维度
        image_array = np.expand_dims(image_array, axis=0)
        
        # 转换为float32
        image_array = image_array.astype(np.float32)
        
        return image_array
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# 批量预处理函数
def batch_preprocess_images(image_data_list):
    """批量预处理图像
    
    Args:
        image_data_list: 图像数据列表（字节流）
    
    Returns:
        预处理后的输入张量列表
    """
    return [preprocess_image(image_data) for image_data in image_data_list]
