"""
测试注册器和接口功能
"""
import unittest
from typing import Dict, Any
import torch
from architect.core.registry import ComponentRegistry
from architect.core.config_manager import ConfigManager
from architect.interfaces.base_model import BaseModel
from architect.interfaces.base_encoder import BaseEncoder
from architect.interfaces.base_head import BaseHead
from architect.unified_model import UniversalVisionModel

class TestRegistryAndInterfaces(unittest.TestCase):
    """测试注册器和接口功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 清空注册器
        ComponentRegistry.clear_registry()
        
        # 注册测试编码器
        @ComponentRegistry.register_backbone('test_encoder')
        def build_test_encoder(config: Dict[str, Any]) -> BaseEncoder:
            """测试编码器构建函数"""
            class TestEncoder(BaseEncoder, torch.nn.Module):
                """测试编码器"""
                def __init__(self, config: Dict[str, Any]):
                    torch.nn.Module.__init__(self)
                    BaseEncoder.__init__(self, config)
                    self.output_dim = config.get('output_dim', 256)
                
                def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                    """提取特征"""
                    return {'features': torch.randn(x.shape[0], self.output_dim)}
                
                def get_output_dim(self) -> int:
                    """获取输出维度"""
                    return self.output_dim
            
            return TestEncoder(config)
        
        # 注册测试分类头
        @ComponentRegistry.register_head('classification')
        def build_test_classification_head(config: Dict[str, Any]) -> BaseHead:
            """测试分类头构建函数"""
            class TestClassificationHead(BaseHead, torch.nn.Module):
                """测试分类头"""
                def __init__(self, config: Dict[str, Any]):
                    torch.nn.Module.__init__(self)
                    BaseHead.__init__(self, config)
                    self.num_classes = config.get('num_classes', 10)
                
                def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
                    """前向传播"""
                    return {'logits': torch.randn(features['features'].shape[0], self.num_classes)}
                
                def compute_loss(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
                    """计算损失"""
                    return torch.tensor(0.0)
                
                def get_task_type(self) -> str:
                    """获取任务类型"""
                    return 'classification'
            
            return TestClassificationHead(config)
        
        # 注册测试检测头
        @ComponentRegistry.register_head('detection')
        def build_test_detection_head(config: Dict[str, Any]) -> BaseHead:
            """测试检测头构建函数"""
            class TestDetectionHead(BaseHead, torch.nn.Module):
                """测试检测头"""
                def __init__(self, config: Dict[str, Any]):
                    torch.nn.Module.__init__(self)
                    BaseHead.__init__(self, config)
                
                def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
                    """前向传播"""
                    return {'boxes': torch.randn(1, 4), 'scores': torch.randn(1)}
                
                def compute_loss(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
                    """计算损失"""
                    return torch.tensor(0.0)
                
                def get_task_type(self) -> str:
                    """获取任务类型"""
                    return 'detection'
            
            return TestDetectionHead(config)
        
        # 触发注册
        build_test_encoder
        build_test_classification_head
        build_test_detection_head
    
    def test_register_backbone(self):
        """测试注册骨干网络"""
        backbones = ComponentRegistry.get_registered_backbones()
        self.assertIn('test_encoder', backbones)
    
    def test_register_head(self):
        """测试注册任务头"""
        heads = ComponentRegistry.get_registered_heads()
        self.assertIn('classification', heads)
        self.assertIn('detection', heads)
    
    def test_get_backbone(self):
        """测试获取骨干网络"""
        config = {'output_dim': 512}
        encoder = ComponentRegistry.get_backbone('test_encoder', config)
        self.assertIsInstance(encoder, BaseEncoder)
        self.assertEqual(encoder.get_output_dim(), 512)
    
    def test_get_head(self):
        """测试获取任务头"""
        config = {'num_classes': 20}
        head = ComponentRegistry.get_head('classification', config)
        self.assertIsInstance(head, BaseHead)
        self.assertEqual(head.get_task_type(), 'classification')
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试获取不存在的骨干网络
        with self.assertRaises(ValueError):
            ComponentRegistry.get_backbone('nonexistent_encoder', {})
        
        # 测试获取不存在的任务头
        with self.assertRaises(ValueError):
            ComponentRegistry.get_head('nonexistent_task', {})
    
    def test_config_manager(self):
        """测试配置管理器"""
        # 创建测试配置
        test_config = {
            'model': {
                'vision_encoder': {
                    'type': 'test_encoder',
                    'output_dim': 256
                },
                'tasks': {
                    'classification': {
                        'enabled': True,
                        'num_classes': 10
                    },
                    'detection': {
                        'enabled': True
                    }
                }
            }
        }
        
        # 验证配置
        try:
            ConfigManager.validate_config(test_config)
            valid = True
        except ValueError:
            valid = False
        self.assertTrue(valid)
    
    def test_universal_model(self):
        """测试通用模型"""
        # 创建测试配置
        test_config = {
            'model': {
                'vision_encoder': {
                    'type': 'test_encoder',
                    'output_dim': 256
                },
                'tasks': {
                    'classification': {
                        'enabled': True,
                        'num_classes': 10
                    },
                    'detection': {
                        'enabled': True
                    }
                }
            }
        }
        
        # 创建模型
        model = UniversalVisionModel(test_config)
        
        # 测试获取支持的任务
        supported_tasks = model.get_supported_tasks()
        self.assertIn('classification', supported_tasks)
        self.assertIn('detection', supported_tasks)
        
        # 测试设置任务
        model.set_task('classification')
        self.assertEqual(model.current_task, 'classification')
        
        # 测试前向传播
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        self.assertIn('task', output)
        self.assertEqual(output['task'], 'classification')
        self.assertIn('logits', output)

if __name__ == '__main__':
    unittest.main()
