import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ...architect.interfaces.base_head import BaseHead
from ...architect.core.registry import ComponentRegistry

@ComponentRegistry.register_head("classification")
class ClassificationHead(BaseHead, nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 2048)
        self.num_classes = config.get('num_classes', 1000)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.dropout_rate = config.get('dropout', 0.1)
        self.use_attention = config.get('use_attention', False)
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.input_dim, 1),
                nn.Tanh(),
                nn.Softmax(dim=1)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None) -> Dict:
        # Extract feature map or sequence
        x = self._extract_features(features)
        
        # Pooling
        if x.dim() == 4: # (B, C, H, W)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
        elif x.dim() == 3: # (B, N, C)
            if self.use_attention:
                weights = self.attention(x) # (B, N, 1)
                x = torch.sum(x * weights, dim=1)
            else:
                x = x.mean(dim=1)
                
        logits = self.classifier(x)
        output = {'logits': logits}
        
        if targets is not None and 'labels' in targets:
            loss = self.compute_loss(output, targets)
            output['loss'] = loss
            output['accuracy'] = self.compute_accuracy(logits, targets['labels'])
            
        return output

    def _extract_features(self, features):
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, dict):
            # Prefer 'p5' or 'last' or the last value
            if 'p5' in features: return features['p5']
            return list(features.values())[-1]
        raise ValueError("Unsupported feature format")

    def compute_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        return self.criterion(predictions['logits'], targets['labels'])
        
    def compute_accuracy(self, logits, labels):
        with torch.no_grad():
            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            return correct.flatten().float().mean()
