import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from ...architect.interfaces.base_head import BaseHead
from ...architect.core.registry import ComponentRegistry

@ComponentRegistry.register_head("detection")
class DetectionHead(BaseHead, nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 80)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_queries = config.get('num_queries', 100)
        
        # Input projection
        self.input_proj = nn.Conv2d(config.get('input_dim', 2048), self.hidden_dim, kernel_size=1)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Embeddings
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Prediction heads
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1) # +1 for background
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4)
        )
        
    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None) -> Dict:
        x = self._extract_features(features)
        bs = x.shape[0]
        
        # Project features
        h = self.input_proj(x) # (B, C, H, W)
        
        # Flatten spatial dimensions
        h = h.flatten(2).permute(2, 0, 1) # (HW, B, C)
        
        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # (N, B, C)
        
        # Decode
        # Note: Ideally we should add positional embeddings to h here
        hs = self.decoder(query_embed, h) # (N, B, C)
        hs = hs.transpose(0, 1) # (B, N, C)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        output = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
        
        if targets is not None and 'boxes' in targets:
            loss_dict = self.compute_loss(output, targets)
            output.update(loss_dict)
            output['loss'] = sum(loss_dict.values())
            
        return output
        
    def _extract_features(self, features):
        if isinstance(features, torch.Tensor): return features
        if isinstance(features, dict):
            # Prefer 'p5' or 'last'
            return list(features.values())[-1]
        raise ValueError("Unsupported feature format")

    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        # Simple greedy matching for demonstration
        # In production, use Hungarian Matcher
        
        pred_logits = predictions['pred_logits']
        pred_boxes = predictions['pred_boxes']
        
        target_classes = targets['labels'] # List of [num_boxes] tensors
        target_boxes = targets['boxes'] # List of [num_boxes, 4] tensors
        
        loss_ce = 0.0
        loss_bbox = 0.0
        
        batch_size = pred_logits.shape[0]
        
        for i in range(batch_size):
            p_logits = pred_logits[i] # (N, num_classes+1)
            p_boxes = pred_boxes[i] # (N, 4)
            t_classes = target_classes[i]
            t_boxes = target_boxes[i]
            
            if len(t_boxes) == 0:
                continue
                
            # Greedy assignment based on L1 cost
            # Expand to (N, M, 4)
            cost = torch.cdist(p_boxes, t_boxes, p=1)
            
            # Simple greedy matching
            matched_indices = []
            used_preds = set()
            
            # Iterate over targets
            for t_idx in range(len(t_boxes)):
                # Find best matching pred that is not used
                best_cost = float('inf')
                best_p_idx = -1
                
                for p_idx in range(self.num_queries):
                    if p_idx in used_preds:
                        continue
                    if cost[p_idx, t_idx] < best_cost:
                        best_cost = cost[p_idx, t_idx]
                        best_p_idx = p_idx
                
                if best_p_idx != -1:
                    matched_indices.append((best_p_idx, t_idx))
                    used_preds.add(best_p_idx)
            
            # Compute losses for matched pairs
            for p_idx, t_idx in matched_indices:
                loss_bbox += F.l1_loss(p_boxes[p_idx], t_boxes[t_idx])
                loss_ce += F.cross_entropy(p_logits[p_idx:p_idx+1], t_classes[t_idx:t_idx+1])
                
            # Background loss for unmatched predictions
            unmatched_mask = torch.ones(self.num_queries, dtype=torch.bool, device=p_logits.device)
            unmatched_mask[list(used_preds)] = False
            # Assume last class is background
            bg_target = torch.full((unmatched_mask.sum(),), self.num_classes, dtype=torch.long, device=p_logits.device)
            loss_ce += F.cross_entropy(p_logits[unmatched_mask], bg_target)
            
        return {
            'loss_ce': loss_ce / batch_size,
            'loss_bbox': loss_bbox / batch_size
        }
