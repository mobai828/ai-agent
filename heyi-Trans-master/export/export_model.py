#!/usr/bin/env python3
"""
模型导出脚本

功能：
- 加载训练好的模型
- 将模型导出为ONNX格式
- 支持不同任务的模型导出
- 验证导出模型的完整性
"""

import argparse
import os
import torch
import yaml

from architect.unified_model import UniversalVisionModel

def export_to_onnx(model, output_path, input_shape=(1, 3, 224, 224), task_name="classification"):
    """将模型导出为ONNX格式
    
    Args:
        model: 要导出的模型
        output_path: 输出ONNX文件路径
        input_shape: 输入张量形状 (batch, channels, height, width)
        task_name: 任务名称
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 定义导出参数
    export_args = {
        'f': output_path,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        'do_constant_folding': True,
        'verbose': False
    }
    
    # 导出模型
    print(f"导出模型到ONNX格式: {output_path}")
    print(f"输入形状: {input_shape}")
    print(f"任务: {task_name}")
    
    # 使用torch.onnx.export导出
    with torch.no_grad():
        torch.onnx.export(
            model=lambda x: model(x, task_name=task_name),
            args=(dummy_input,),
            **export_args
        )
    
    print(f"模型导出成功: {output_path}")
    return output_path

def validate_onnx_model(onnx_path):
    """验证ONNX模型的完整性
    
    Args:
        onnx_path: ONNX模型文件路径
    
    Returns:
        是否验证成功
    """
    try:
        import onnx
        import onnxruntime
        
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        
        # 检查模型结构
        onnx.checker.check_model(model)
        print("ONNX模型结构验证通过")
        
        # 使用ONNX Runtime验证模型可以正常加载
        session = onnxruntime.InferenceSession(onnx_path)
        print("ONNX模型加载验证通过")
        
        # 打印模型信息
        print(f"模型输入: {[input.name for input in session.get_inputs()]}")
        print(f"模型输出: {[output.name for output in session.get_outputs()]}")
        
        return True
    except ImportError:
        print("警告: 缺少ONNX或ONNX Runtime，无法验证模型")
        return False
    except Exception as e:
        print(f"错误: ONNX模型验证失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型导出脚本')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--output_dir', type=str, default='./export/models',
                        help='导出模型输出目录')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx'],
                        help='导出格式')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'detection', 'segmentation', 'retrieval'],
                        help='导出模型的任务')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 224, 224),
                        help='输入张量形状')
    parser.add_argument('--validate', action='store_true',
                        help='验证导出模型的完整性')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置文件
    print(f"加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    print("初始化模型...")
    model = UniversalVisionModel(config)
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # 构建输出文件名
    output_filename = f"model_{args.task}.onnx"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 导出模型
    if args.format == 'onnx':
        export_to_onnx(model, output_path, args.input_shape, args.task)
    else:
        raise ValueError(f"不支持的导出格式: {args.format}")
    
    # 验证导出模型
    if args.validate:
        validate_onnx_model(output_path)
    
    print(f"\n===== 导出完成 =====")
    print(f"导出模型路径: {output_path}")
    print(f"任务: {args.task}")
    print(f"输入形状: {args.input_shape}")

if __name__ == "__main__":
    main()
