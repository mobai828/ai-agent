import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import logging

class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, task_type, transform=None):
        self.root_dir = root_dir
        self.task_type = task_type
        self.transform = transform
        self.data = []
        
        # 检查目录是否存在
        if not os.path.exists(root_dir):
            logging.warning(f"数据集目录不存在: {root_dir}，将使用空数据集")
            return
        
        # 根据任务类型加载数据
        try:
            if task_type == 'classification':
                # 假设目录结构为: root_dir/class_name/image.jpg
                for class_name in os.listdir(root_dir):
                    class_path = os.path.join(root_dir, class_name)
                    if os.path.isdir(class_path):
                        for img_name in os.listdir(class_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                self.data.append({
                                    'image_path': os.path.join(class_path, img_name),
                                    'label': class_name
                                })
            elif task_type == 'retrieval':
                # 假设目录结构与分类相同，使用类别作为检索标签
                for class_name in os.listdir(root_dir):
                    class_path = os.path.join(root_dir, class_name)
                    if os.path.isdir(class_path):
                        for img_name in os.listdir(class_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                self.data.append({
                                    'image_path': os.path.join(class_path, img_name),
                                    'label': class_name
                                })
            elif task_type == 'detection':
                # 假设使用COCO格式的标注文件
                annotations_file = os.path.join(root_dir, 'annotations.json')
                if os.path.exists(annotations_file):
                    with open(annotations_file, 'r') as f:
                        annotations = json.load(f)
                    # 处理标注数据
                    # 这里简化处理，实际需要根据COCO格式解析
                else:
                    logging.warning(f"检测任务的标注文件不存在: {annotations_file}")
            elif task_type == 'segmentation':
                # 假设使用PASCAL VOC格式
                # 图像目录和标签目录
                img_dir = os.path.join(root_dir, 'JPEGImages')
                seg_dir = os.path.join(root_dir, 'SegmentationClass')
                if os.path.exists(img_dir) and os.path.exists(seg_dir):
                    for img_name in os.listdir(img_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            seg_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                            self.data.append({
                                'image_path': os.path.join(img_dir, img_name),
                                'mask_path': os.path.join(seg_dir, seg_name)
                            })
                else:
                    logging.warning(f"分割任务的目录不存在: img_dir={img_dir}, seg_dir={seg_dir}")
        except Exception as e:
            logging.error(f"加载数据集时出错: {str(e)}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        
        # 根据任务类型处理标签
        if self.task_type == 'classification' or self.task_type == 'retrieval':
            # 假设标签是字符串，需要映射到数字
            # 这里简化处理，实际需要创建标签映射
            label = hash(item['label']) % 1000  # 临时处理
            target = {'labels': torch.tensor(label)}
        elif self.task_type == 'detection':
            # 这里简化处理，实际需要加载边界框和类别
            target = {'boxes': torch.tensor([[0, 0, 100, 100]]), 'labels': torch.tensor([0])}
        elif self.task_type == 'segmentation':
            # 加载分割掩码
            mask = Image.open(item['mask_path']).convert('L')
            target = {'masks': torch.tensor(mask)}
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'target': target}
