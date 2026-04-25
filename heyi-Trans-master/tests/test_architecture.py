"""
架构测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from architect.core.config_manager import ConfigManager

class TestArchitecture(unittest.TestCase):
    
    def test_config_loading(self):
        """测试配置加载"""
        config = ConfigManager.load_config("configs/base.yaml")
        self.assertIn('project', config)
        self.assertIn('model', config)
        self.assertEqual(config['project']['name'], 'vision_thransformer')
    
    def test_h20_config(self):
        """测试H20配置"""
        config = ConfigManager.load_config("configs/h20_config.yaml")
        self.assertIn('hardware', config)
        self.assertEqual(config['hardware']['device'], 'cuda:0')
        self.assertEqual(config['training']['batch_size'], 4)

if __name__ == '__main__':
    unittest.main()
