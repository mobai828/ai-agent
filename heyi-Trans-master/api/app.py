#!/usr/bin/env python3
"""
API服务主文件

功能：
- 提供模型推理接口
- 处理HTTP请求和响应
- 支持多种任务的模型推理
- 实现服务健康检查
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
from datetime import datetime

from api.model_handler import ModelHandler
from api.preprocessing import preprocess_image
from api.config import API_CONFIG

# 创建FastAPI应用
app = FastAPI(
    title="Transformer Model API",
    description="提供Transformer模型的推理接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型处理器实例
model_handler = None

# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查接口"""
    global model_handler
    status = "healthy"
    models_loaded = []
    
    if model_handler and model_handler.models:
        models_loaded = list(model_handler.models.keys())
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "service": "Transformer Model API"
    }

# 模型信息接口
@app.get("/models")
async def get_models():
    """获取加载的模型信息"""
    global model_handler
    
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    models_info = {}
    for task, model_info in model_handler.models.items():
        models_info[task] = {
            "model_path": model_info["model_path"],
            "loaded_at": model_info["loaded_at"].isoformat(),
            "task": task
        }
    
    return {
        "models": models_info,
        "total_models": len(models_info)
    }

# 推理请求模型
class InferenceRequest(BaseModel):
    """推理请求模型"""
    task: str = "classification"
    # 注意：对于图像输入，我们使用文件上传而不是JSON

# 推理接口（文件上传）
@app.post("/inference")
async def inference(
    task: str = "classification",
    file: UploadFile = File(...)
):
    """模型推理接口
    
    Args:
        task: 任务类型 (classification, detection, segmentation, retrieval)
        file: 上传的图像文件
    
    Returns:
        推理结果
    """
    global model_handler
    
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    # 检查任务是否支持
    if task not in model_handler.models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: {task}. Supported tasks: {list(model_handler.models.keys())}"
        )
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 读取上传的文件
        image_data = await file.read()
        
        # 预处理图像
        input_tensor = preprocess_image(image_data)
        
        # 执行推理
        result = model_handler.infer(task, input_tensor)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 构建响应
        response = {
            "task": task,
            "result": result,
            "inference_time_ms": round(inference_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# 批量推理接口
@app.post("/batch_inference")
async def batch_inference(
    task: str = "classification",
    files: list[UploadFile] = File(...)
):
    """批量模型推理接口
    
    Args:
        task: 任务类型
        files: 上传的图像文件列表
    
    Returns:
        批量推理结果
    """
    global model_handler
    
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    # 检查任务是否支持
    if task not in model_handler.models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: {task}. Supported tasks: {list(model_handler.models.keys())}"
        )
    
    # 限制批量大小
    if len(files) > API_CONFIG["batch_size_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit. Maximum batch size: {API_CONFIG['batch_size_limit']}"
        )
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        results = []
        
        for file in files:
            # 读取上传的文件
            image_data = await file.read()
            
            # 预处理图像
            input_tensor = preprocess_image(image_data)
            
            # 执行推理
            result = model_handler.infer(task, input_tensor)
            
            results.append({
                "filename": file.filename,
                "result": result
            })
        
        # 计算推理时间
        total_time = (time.time() - start_time) * 1000  # 转换为毫秒
        avg_time_per_image = total_time / len(files)
        
        # 构建响应
        response = {
            "task": task,
            "results": results,
            "total_images": len(results),
            "total_time_ms": round(total_time, 2),
            "avg_time_per_image_ms": round(avg_time_per_image, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

# 服务启动事件
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型处理器"""
    global model_handler
    print("Initializing model handler...")
    
    try:
        # 初始化模型处理器
        model_handler = ModelHandler()
        
        # 加载配置的模型
        for task, model_path in API_CONFIG["models"].items():
            if model_path:
                try:
                    model_handler.load_model(task, model_path)
                    print(f"Loaded model for task: {task}")
                except Exception as e:
                    print(f"Failed to load model for task {task}: {str(e)}")
        
        print("Model handler initialized successfully")
    except Exception as e:
        print(f"Failed to initialize model handler: {str(e)}")

# 服务关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    global model_handler
    print("Shutting down model handler...")
    
    if model_handler:
        model_handler.unload_all_models()
        print("All models unloaded")

# 主函数
if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        workers=API_CONFIG["workers"]
    )
