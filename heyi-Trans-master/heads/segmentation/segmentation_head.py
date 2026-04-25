import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ...architect.interfaces.base_head import BaseHead
from ...architect.core.registry import ComponentRegistry

@ComponentRegistry.register_head("segmentation")
class SegmentationHead(BaseHead, nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 21)
        self.input_dim = config.get('input_dim', 2048)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Simple FCN Head
        self.block = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.hidden_dim, self.num_classes, 1)
        )
        
    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None) -> Dict:
        x = self._extract_features(features)
        
        logits = self.block(x)
        
        output = {'logits': logits}
        
        if targets is not None and 'masks' in targets:
            target_masks = targets['masks']
            
            # Upsample logits to match target size
            if logits.shape[-2:] != target_masks.shape[-2:]:
                logits = F.interpolate(logits, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
                output['logits'] = logits
            
            loss = self.compute_loss(output, targets)
            output['loss'] = loss
            output['mIoU'] = self.compute_iou(logits, target_masks)
            
        return output

    def _extract_features(self, features):
        if isinstance(features, torch.Tensor): return features
        if isinstance(features, dict): return list(features.values())[-1]
        raise ValueError("Unsupported feature format")

    def compute_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        logits = predictions['logits']
        masks = targets['masks']
        
        # Cross Entropy
        ce_loss = F.cross_entropy(logits, masks, ignore_index=255)
        
        # Dice Loss
        dice_loss = self.dice_loss(logits, masks)
        
        return ce_loss + dice_loss

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        
        # Create one-hot targets
        # Ensure targets are within range [0, num_classes-1] for one_hot
        # Mask out ignore_index (usually 255)
        valid_mask = (targets != 255)
        targets_masked = targets.clone()
        targets_masked[~valid_mask] = 0 # Avoid index out of bound
        
        targets_one_hot = F.one_hot(targets_masked, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Zero out ignored regions in one-hot
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def compute_iou(self, logits, masks):
        pred = torch.argmax(logits, dim=1)
        ious = []
        # Move to CPU for loop to avoid sync overhead if many classes, 
        # but for training metrics typically we do it on GPU or just a few classes
        for cls in range(self.num_classes):
            pred_mask = pred == cls
            target_mask = masks == cls
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            if union > 0:
                ious.append(intersection / union)
        
        return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=logits.device)
