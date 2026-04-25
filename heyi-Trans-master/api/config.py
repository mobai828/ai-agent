#!/usr/bin/env python3
"""
API服务配置文件
"""

# API服务配置
API_CONFIG = {
    # 服务配置
    "host": "0.0.0.0",  # 监听所有网络接口
    "port": 8000,       # 服务端口
    "reload": False,    # 开发模式下可设置为True
    "workers": 4,       # 工作进程数
    
    # 模型配置
    "models": {
        "classification": "export/models/model_classification.onnx",
        "detection": "export/models/model_detection.onnx",
        "segmentation": "export/models/model_segmentation.onnx",
        "retrieval": "export/models/model_retrieval.onnx"
    },
    
    # 批量处理配置
    "batch_size_limit": 16,  # 最大批量处理大小
    
    # 超时配置
    "timeout": 30,  # 推理超时时间（秒）
    
    # 日志配置
    "log_level": "info",
    "log_file": "api/logs/api.log"
}
