import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ...architect.interfaces.base_head import BaseHead
from ...architect.core.registry import ComponentRegistry

@ComponentRegistry.register_head("retrieval")
class RetrievalHead(BaseHead, nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 2048)
        self.output_dim = config.get('output_dim', 128)
        self.margin = config.get('margin', 1.0)
        
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )
        
    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None) -> Dict:
        x = self._extract_features(features)
        
        # Global Pooling if needed
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
        elif x.dim() == 3:
            x = x.mean(dim=1)
            
        embeddings = self.projector(x)
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        output = {'embeddings': embeddings}
        
        if targets is not None and 'labels' in targets:
            loss = self.compute_loss(output, targets)
            output['loss'] = loss
            
        return output

    def _extract_features(self, features):
        if isinstance(features, torch.Tensor): return features
        if isinstance(features, dict): return list(features.values())[-1]
        raise ValueError("Unsupported feature format")

    def compute_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        embeddings = predictions['embeddings']
        labels = targets['labels']
        
        # Batch Hard Triplet Loss
        
        # Compute pairwise distances
        # dist_mat = ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # Since embeddings are normalized, ||a||^2 = 1
        # dist_mat = 2 - 2<a, b>
        # But cdist computes euclidean distance directly
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        # Mask for positives (same label)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Hardest positive: max distance among same class
        # We assume there is at least one positive (itself)
        hardest_pos_dist = (dist_mat * labels_eq.float()).max(dim=1)[0]
        
        # Hardest negative: min distance among different class
        # Add large value to same class to ignore them in min
        # Ensure we have at least one negative, otherwise loss is 0 for that anchor
        # If batch size is small or only one class, this might be issue.
        # Assuming batch has negatives.
        
        # Use a large value slightly smaller than float('inf') to be safe
        large_val = 1e6
        dist_mat_neg = dist_mat + labels_eq.float() * large_val
        hardest_neg_dist = dist_mat_neg.min(dim=1)[0]
        
        # Handle cases where there are no negatives (e.g. all same class)
        # hardest_neg_dist will be > large_val/2
        valid_negatives = hardest_neg_dist < large_val / 2
        
        if valid_negatives.sum() > 0:
            loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            loss = loss[valid_negatives].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss
