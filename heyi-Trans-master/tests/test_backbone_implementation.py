import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.encoders import ResNetEncoder, ViTEncoder
from backbone.fusion import FeatureAdapter

class TestBackbone(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on {self.device}")
        
    def test_resnet_encoder(self):
        print("\nTesting ResNet Encoder...")
        config = {'depth': 50, 'pretrained': False} # Use False for speed/no download
        model = ResNetEncoder(config).to(self.device)
        
        # Dummy input: Batch=2, C=3, H=224, W=224
        x = torch.randn(2, 3, 224, 224).to(self.device)
        features = model(x)
        
        print("ResNet features keys:", features.keys())
        self.assertIn('c2', features)
        self.assertIn('c5', features)
        
        # Check shapes
        # c2: stride 4 -> 224/4 = 56
        self.assertEqual(features['c2'].shape[-2:], (56, 56))
        # c5: stride 32 -> 224/32 = 7
        self.assertEqual(features['c5'].shape[-2:], (7, 7))
        
    def test_vit_encoder(self):
        print("\nTesting ViT Encoder...")
        config = {'model_name': 'vit_b_16', 'pretrained': False}
        model = ViTEncoder(config).to(self.device)
        
        x = torch.randn(2, 3, 224, 224).to(self.device)
        features = model(x)
        
        print("ViT features keys:", features.keys())
        self.assertIn('cls_token', features)
        self.assertIn('patch_features', features)
        
        # ViT-B/16: 224/16 = 14 -> 14x14 = 196 patches
        self.assertEqual(features['patch_features'].shape[1], 196)
        self.assertEqual(features['cls_token'].shape[1], 768) # Hidden dim
        
    def test_adapter_resnet(self):
        print("\nTesting Adapter with ResNet...")
        # Mock ResNet features
        features = {
            'c2': torch.randn(2, 256, 56, 56),
            'c3': torch.randn(2, 512, 28, 28),
            'c4': torch.randn(2, 1024, 14, 14),
            'c5': torch.randn(2, 2048, 7, 7)
        }
        
        config = {
            'backbone_type': 'resnet',
            'out_channels': 256,
            'in_channels': {'c2': 256, 'c3': 512, 'c4': 1024, 'c5': 2048}
        }
        
        adapter = FeatureAdapter(config)
        outs = adapter(features)
        
        print("Adapter (ResNet) output keys:", outs.keys())
        for k, v in outs.items():
            self.assertEqual(v.shape[1], 256)
            print(f"{k} shape: {v.shape}")
            
    def test_adapter_vit(self):
        print("\nTesting Adapter with ViT...")
        # Mock ViT features: [B, N, D]
        # 14x14 patches = 196
        features = {
            'patch_features': torch.randn(2, 196, 768)
        }
        
        config = {
            'backbone_type': 'vit',
            'out_channels': 256,
            'in_channels': 768
        }
        
        adapter = FeatureAdapter(config)
        outs = adapter(features)
        
        print("Adapter (ViT) output keys:", outs.keys())
        # Check shapes
        # c4 (base 1/16): 14x14
        self.assertEqual(outs['c4'].shape[-2:], (14, 14))
        # c3 (up 1/8): 28x28
        self.assertEqual(outs['c3'].shape[-2:], (28, 28))
        # c2 (up 1/4): 56x56
        self.assertEqual(outs['c2'].shape[-2:], (56, 56))
        # c5 (down 1/32): 7x7
        self.assertEqual(outs['c5'].shape[-2:], (7, 7))

if __name__ == '__main__':
    unittest.main()
