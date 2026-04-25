#!/usr/bin/env python3
"""
自动化评估脚本

功能：
- 读取评估配置
- 自动执行模型评估
- 生成评估报告
- 支持定期执行
"""

import os
import subprocess
import yaml
from datetime import datetime

def load_config(config_path):
    """加载评估配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_evaluation(config):
    """执行模型评估
    
    Args:
        config: 评估配置
    
    Returns:
        评估结果路径
    """
    eval_config = config['evaluation']
    
    # 构建评估命令
    cmd = [
        'python', 'evaluation/evaluate.py',
        '--config', eval_config['model']['config_path'],
        '--model_path', eval_config['model']['weights_path'],
        '--data_dir', eval_config['data']['data_dir'],
        '--batch_size', str(eval_config['data']['batch_size']),
        '--output_dir', eval_config['output']['reports_dir'],
        '--device', eval_config['settings']['device']
    ]
    
    print(f"执行评估命令: {' '.join(cmd)}")
    
    # 执行评估
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 打印输出
    print("评估输出:")
    print(result.stdout)
    
    if result.stderr:
        print("评估错误:")
        print(result.stderr)
    
    # 检查执行状态
    if result.returncode != 0:
        raise RuntimeError(f"评估执行失败，返回码: {result.returncode}")
    
    # 返回评估报告路径
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_dir = eval_config['output']['reports_dir']
    return report_dir

def main():
    """主函数"""
    # 加载配置
    config_path = 'evaluation/config.yaml'
    config = load_config(config_path)
    
    print("===== 开始自动化评估 =====")
    print(f"评估时间: {datetime.now().isoformat()}")
    print(f"模型配置: {config['evaluation']['model']['config_path']}")
    print(f"模型权重: {config['evaluation']['model']['weights_path']}")
    print(f"测试数据: {config['evaluation']['data']['data_dir']}")
    print(f"设备: {config['evaluation']['settings']['device']}")
    
    try:
        # 执行评估
        report_path = run_evaluation(config)
        
        print(f"\n===== 评估完成 =====")
        print(f"评估报告已生成到: {report_path}")
        print(f"评估时间: {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"\n===== 评估失败 =====")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main()
