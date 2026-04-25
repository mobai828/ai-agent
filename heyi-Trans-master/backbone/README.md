# Backbone Network Documentation

## Overview
This module contains the backbone networks for feature extraction and the feature adapter for aligning outputs.

## Encoders

### ResNetEncoder
- **Path**: `backbone/encoders/resnet_encoder.py`
- **Supported Models**: ResNet50, ResNet101
- **Output Format**: Dictionary with keys `c2`, `c3`, `c4`, `c5`.
- **Feature Dimensions**:
  - `c2`: 256 channels (Stride 4)
  - `c3`: 512 channels (Stride 8)
  - `c4`: 1024 channels (Stride 16)
  - `c5`: 2048 channels (Stride 32)

### ViTEncoder
- **Path**: `backbone/encoders/vit_encoder.py`
- **Supported Models**: ViT-B/16, ViT-L/16
- **Output Format**: Dictionary with keys `cls_token`, `patch_features`.
- **Feature Dimensions**:
  - `vit_b_16`: 768 channels
  - `vit_l_16`: 1024 channels

## Feature Adapter

### FeatureAdapter
- **Path**: `backbone/fusion/feature_adapter.py`
- **Function**: Unifies outputs from different backbones into a standard FPN format.
- **Output Format**: Dictionary with keys `c2`, `c3`, `c4`, `c5`.
- **Output Dimension**: Configurable (default 256).

## Usage Example

```python
from backbone import ResNetEncoder, FeatureAdapter

# ResNet
encoder = ResNetEncoder({'depth': 50})
features = encoder(image)

# Adapter
adapter = FeatureAdapter({
    'backbone_type': 'resnet',
    'in_channels': encoder.get_feature_dims(),
    'out_channels': 256
})
fpn_features = adapter(features)
```
