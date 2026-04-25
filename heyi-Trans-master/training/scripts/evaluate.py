import sys
import os

# 添加当前目录到Python搜索路径
sys.path.insert(0, os.path.abspath('.'))

import torch
from architect.unified_model import UniversalVisionModel
from architect.core.config_manager import ConfigManager
from training.data.dataset import MultiTaskDataset
from training.data.transforms import get_transforms
from torch.utils.data import DataLoader
import argparse
import logging

# 导入backbone模块，确保ViT编码器被注册
import backbone.encoders

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(config_path, checkpoint_path):
    # 加载配置
    config = ConfigManager.load_config(config_path)
    
    # 创建模型
    model = UniversalVisionModel(config)
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Model loaded from {checkpoint_path}")
    
    # 数据加载
    val_datasets = {}
    val_loaders = {}
    
    for task_name in model.get_supported_tasks():
        if task_name not in config['data']:
            logging.warning(f"Task {task_name} not found in data config, skipping")
            continue
            
        val_datasets[task_name] = MultiTaskDataset(
            root_dir=config['data'][task_name]['val_root'],
            task_type=task_name,
            transform=get_transforms(task_name, is_train=False)
        )
        val_loaders[task_name] = DataLoader(
            val_datasets[task_name],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )
        logging.info(f"Loaded {len(val_datasets[task_name])} val samples for {task_name}")
    
    # 评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    metrics = {}
    with torch.no_grad():
        for task_name in val_loaders:
            task_metrics = []
            for batch in val_loaders[task_name]:
                images = batch['image'].to(device)
                targets = {k: v.to(device) for k, v in batch['target'].items()}
                
                model.set_task(task_name)
                outputs = model(images, task_name=task_name, targets=targets)
                
                # 计算指标
                if task_name == 'classification':
                    if 'accuracy' in outputs:
                        task_metrics.append(outputs['accuracy'].item())
                elif task_name == 'segmentation':
                    if 'mIoU' in outputs:
                        task_metrics.append(outputs['mIoU'].item())
                # 其他任务的指标计算
            
            if task_metrics:
                metrics[task_name] = sum(task_metrics) / len(task_metrics)
            else:
                metrics[task_name] = 0.0
    
    logging.info("Evaluation Results:")
    for task_name, metric in metrics.items():
        logging.info(f"{task_name}: {metric:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
