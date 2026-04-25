import sys
import os

# 添加当前目录到Python搜索路径
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from architect.unified_model import UniversalVisionModel
from architect.core.config_manager import ConfigManager
from training.data.dataset import MultiTaskDataset
from training.data.transforms import get_transforms
import argparse
import logging

# 导入backbone模块，确保ViT编码器被注册
import backbone.encoders

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path):
    # 加载配置
    config = ConfigManager.load_config(config_path)
    
    # 创建模型
    model = UniversalVisionModel(config)
    logging.info(f"Model created with tasks: {model.get_supported_tasks()}")
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 数据加载
    train_datasets = {}
    val_datasets = {}
    train_loaders = {}
    val_loaders = {}
    
    for task_name in model.get_supported_tasks():
        if task_name not in config['data']:
            logging.warning(f"Task {task_name} not found in data config, skipping")
            continue
            
        try:
            train_datasets[task_name] = MultiTaskDataset(
                root_dir=config['data'][task_name]['train_root'],
                task_type=task_name,
                transform=get_transforms(task_name, is_train=True)
            )
            val_datasets[task_name] = MultiTaskDataset(
                root_dir=config['data'][task_name]['val_root'],
                task_type=task_name,
                transform=get_transforms(task_name, is_train=False)
            )
            
            # 检查数据集是否为空
            if len(train_datasets[task_name]) == 0:
                logging.warning(f"训练数据集为空: {task_name}")
                continue
            if len(val_datasets[task_name]) == 0:
                logging.warning(f"验证数据集为空: {task_name}")
                continue
            
            train_loaders[task_name] = DataLoader(
                train_datasets[task_name],
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training']['num_workers']
            )
            val_loaders[task_name] = DataLoader(
                val_datasets[task_name],
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers']
            )
            logging.info(f"Loaded {len(train_datasets[task_name])} train and {len(val_datasets[task_name])} val samples for {task_name}")
        except Exception as e:
            logging.error(f"加载{task_name}任务的数据集时出错: {str(e)}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['max_epochs']
    )
    
    # 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(f"Training on device: {device}")
    
    # 检查是否有可用的训练加载器
    if not train_loaders:
        logging.error("没有可用的训练加载器，请检查数据集配置")
        return
    
    for epoch in range(config['training']['max_epochs']):
        model.train()
        total_loss = 0
        
        # 多任务训练
        for task_name in train_loaders:
            try:
                for batch_idx, batch in enumerate(train_loaders[task_name]):
                    # 准备数据
                    images = batch['image'].to(device)
                    targets = {k: v.to(device) for k, v in batch['target'].items()}
                    
                    # 设置任务
                    model.set_task(task_name)
                    
                    # 前向传播
                    outputs = model(images, task_name=task_name, targets=targets)
                    
                    # 计算损失
                    loss = outputs['loss']
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 打印进度
                    if batch_idx % 100 == 0:
                        logging.info(f"Task {task_name}, Epoch {epoch+1}/{config['training']['max_epochs']}, "
                                     f"Batch {batch_idx}/{len(train_loaders[task_name])}, "
                                     f"Loss: {loss.item():.4f}")
            except Exception as e:
                logging.error(f"训练{task_name}任务时出错: {str(e)}")
        
        # 学习率更新
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for task_name in val_loaders:
                try:
                    for batch in val_loaders[task_name]:
                        images = batch['image'].to(device)
                        targets = {k: v.to(device) for k, v in batch['target'].items()}
                        
                        model.set_task(task_name)
                        outputs = model(images, task_name=task_name, targets=targets)
                        val_loss += outputs['loss'].item()
                except Exception as e:
                    logging.error(f"验证{task_name}任务时出错: {str(e)}")
        
        # 日志
        logging.info(f"Epoch {epoch+1}/{config['training']['max_epochs']}, "
                     f"Train Loss: {total_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % config['training']['save_interval'] == 0:
            try:
                checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Model saved to {checkpoint_path}")
            except Exception as e:
                logging.error(f"保存模型时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
