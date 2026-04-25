# 模型版本管理文档

## 1. 版本管理规范

### 1.1 版本号格式

采用语义化版本号格式：`MAJOR.MINOR.PATCH`

- **MAJOR**: 模型架构或核心算法发生重大变化
- **MINOR**: 添加新功能或改进现有功能
- **PATCH**: 修复bug或进行小的优化

### 1.2 模型命名规范

```
model_{task}_{version}.onnx
```

示例：
- `model_classification_1.0.0.onnx`
- `model_detection_1.1.0.onnx`

## 2. 导出流程

### 2.1 准备工作

1. 确保模型训练完成并保存了权重文件
2. 验证模型在测试集上的性能
3. 准备导出配置

### 2.2 导出步骤

1. 运行导出脚本：
   ```bash
   python export/export_model.py --config configs/base.yaml --model_path models/best_model.pth --task classification --validate
   ```

2. 验证导出模型：
   - 检查ONNX文件是否生成
   - 验证模型结构完整性
   - 测试模型推理性能

3. 记录版本信息

## 3. 版本记录

| 版本号 | 导出日期 | 任务类型 | 模型路径 | 性能指标 | 备注 |
|--------|----------|----------|----------|----------|------|
| 1.0.0 | 2026-02-24 | classification | export/models/model_classification_1.0.0.onnx | 准确率: 0.92 | 初始版本 |
| 1.0.0 | 2026-02-24 | detection | export/models/model_detection_1.0.0.onnx | mAP: 0.85 | 初始版本 |
| 1.0.0 | 2026-02-24 | segmentation | export/models/model_segmentation_1.0.0.onnx | mIoU: 0.78 | 初始版本 |
| 1.0.0 | 2026-02-24 | retrieval | export/models/model_retrieval_1.0.0.onnx | Recall@1: 0.65 | 初始版本 |

## 4. 模型部署指南

### 4.1 环境要求

- Python 3.8+
- ONNX Runtime 1.10.0+
- NumPy
- Pillow

### 4.2 部署步骤

1. 安装依赖：
   ```bash
   pip install onnxruntime numpy pillow
   ```

2. 加载模型：
   ```python
   import onnxruntime
   import numpy as np
   
   # 加载模型
   session = onnxruntime.InferenceSession('export/models/model_classification_1.0.0.onnx')
   
   # 准备输入
   input_name = session.get_inputs()[0].name
   output_name = session.get_outputs()[0].name
   ```

3. 模型推理：
   ```python
   # 预处理输入图像
   # 执行推理
   outputs = session.run([output_name], {input_name: input_data})
   ```

## 5. 性能优化

### 5.1 模型优化策略

1. **量化**：将模型权重从FP32量化为INT8
2. **裁剪**：移除不必要的模型层
3. **批处理**：使用批处理提高推理效率

### 5.2 部署优化

1. 使用ONNX Runtime的优化选项
2. 配置适当的执行提供者（CPU/GPU）
3. 启用内存优化

## 6. 模型更新流程

1. 训练新模型
2. 评估新模型性能
3. 导出新模型
4. 记录版本信息
5. 部署新模型
6. 监控部署后性能

## 7. 故障排除

### 7.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型加载失败 | ONNX文件损坏或版本不兼容 | 重新导出模型，确保使用兼容的ONNX版本 |
| 推理速度慢 | 模型未优化或硬件资源不足 | 优化模型，使用适当的硬件加速 |
| 推理结果错误 | 输入预处理不正确 | 检查输入预处理步骤，确保与训练时一致 |

### 7.2 日志记录

部署过程中应记录以下信息：
- 模型版本
- 部署时间
- 硬件环境
- 推理性能指标
- 错误日志

## 8. 安全考虑

1. 模型文件应妥善保管，避免未授权访问
2. 定期更新模型以修复潜在的安全问题
3. 监控模型部署环境的安全性

## 9. 未来计划

- 支持更多导出格式（如TensorRT、TFLite）
- 实现自动模型版本管理
- 开发模型性能监控系统
