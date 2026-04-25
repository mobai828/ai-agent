#!/usr/bin/env python3
"""
模型处理器

功能：
- 加载和管理不同任务的模型
- 执行模型推理
- 处理模型加载和卸载
"""

import os
import onnxruntime
import numpy as np
from datetime import datetime

class ModelHandler:
    """模型处理器类"""
    
    def __init__(self):
        """初始化模型处理器"""
        self.models = {}
        self.sessions = {}
    
    def load_model(self, task, model_path):
        """加载模型
        
        Args:
            task: 任务类型
            model_path: 模型文件路径
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # 创建ONNX Runtime会话
            session = onnxruntime.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']  # 可以添加GPU提供者
            )
            
            # 保存模型信息
            self.models[task] = {
                "model_path": model_path,
                "loaded_at": datetime.now(),
                "task": task
            }
            
            # 保存会话
            self.sessions[task] = session
            
            print(f"Model loaded successfully for task: {task}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def unload_model(self, task):
        """卸载模型
        
        Args:
            task: 任务类型
        """
        if task in self.models:
            del self.models[task]
        
        if task in self.sessions:
            del self.sessions[task]
        
        print(f"Model unloaded for task: {task}")
    
    def unload_all_models(self):
        """卸载所有模型"""
        tasks = list(self.models.keys())
        for task in tasks:
            self.unload_model(task)
        
        print("All models unloaded")
    
    def infer(self, task, input_tensor):
        """执行模型推理
        
        Args:
            task: 任务类型
            input_tensor: 输入张量
        
        Returns:
            推理结果
        
        Raises:
            ValueError: 如果任务不存在或模型未加载
            RuntimeError: 如果推理失败
        """
        # 检查任务是否存在
        if task not in self.sessions:
            raise ValueError(f"Model not loaded for task: {task}")
        
        try:
            # 获取会话
            session = self.sessions[task]
            
            # 获取输入和输出名称
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # 执行推理
            outputs = session.run([output_name], {input_name: input_tensor})
            
            # 处理推理结果
            result = self._process_result(task, outputs[0])
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _process_result(self, task, output):
        """处理推理结果
        
        Args:
            task: 任务类型
            output: 模型输出
        
        Returns:
            处理后的结果
        """
        # 根据任务类型处理结果
        if task == "classification":
            # 分类任务：返回概率最高的类别
            probabilities = self._softmax(output)
            top_indices = np.argsort(probabilities)[::-1][:5]  # 前5个类别
            top_scores = probabilities[top_indices].tolist()
            
            return {
                "class_indices": top_indices.tolist(),
                "scores": top_scores,
                "task": task
            }
        
        elif task == "detection":
            # 检测任务：返回边界框和类别
            # 简化处理，实际应根据模型输出格式进行解析
            return {
                "detections": [],
                "task": task
            }
        
        elif task == "segmentation":
            # 分割任务：返回分割掩码
            return {
                "segmentation": output.tolist(),
                "task": task
            }
        
        elif task == "retrieval":
            # 检索任务：返回特征向量
            return {
                "embedding": output.flatten().tolist(),
                "task": task
            }
        
        else:
            # 默认处理：返回原始输出
            return {
                "output": output.tolist(),
                "task": task
            }
    
    def _softmax(self, x):
        """计算softmax
        
        Args:
            x: 输入数组
        
        Returns:
            softmax结果
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)
