#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型模块
包含各种用于视频特征提取的预训练模型
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 检查PytorchVideo可用性
try:
    from pytorchvideo.models.hub import i3d_r50
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    logger.warning("PytorchVideo not available")


class I3DFeatureExtractorRGB(nn.Module):
    """I3D RGB模态特征提取器，提取分类层前的特征"""
    
    feature_dim = 2048  # I3D特征维度
    
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("PytorchVideo is required for I3D model")
        
        # 加载完整的I3D模型
        self.full_model = i3d_r50(pretrained=pretrained)
        
        # 如果提供了自定义权重，加载它们
        if weights_path:
            self._load_custom_weights(weights_path)
        
        self.features = None
        self._register_hooks()
        
        logger.info(f"I3D RGB feature extractor initialized (pretrained={pretrained})")
    
    def _load_custom_weights(self, weights_path: str):
        """加载自定义权重"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.full_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom weights from {weights_path}: {e}")
    
    def _register_hooks(self):
        """注册hooks来提取分类层前的特征"""
        def feature_hook(module, input, output):
            # 保存特征（在全连接层之前，AvgPool3d之后）
            if len(output.shape) == 2:  # [batch_size, feature_dim]
                self.features = output
            elif len(output.shape) > 2:
                # 如果是多维输出，应用全局平均池化
                dims_to_pool = list(range(2, len(output.shape)))
                self.features = torch.mean(output, dim=dims_to_pool)
        
        # 精确定位到 blocks.6.pool 层
        hook_registered = False
        for name, module in self.full_model.named_modules():
            if name == 'blocks.6.pool':
                module.register_forward_hook(feature_hook)
                logger.info(f"Registered hook at I3D feature extraction layer: {name}")
                hook_registered = True
                break
        
        if not hook_registered:
            raise RuntimeError("Failed to register hook for I3D feature extraction layer")
    
    def forward(self, x):
        """
        前向传播提取特征
        
        Args:
            x: 输入视频张量 [batch_size, channels, frames, height, width]
            
        Returns:
            特征张量 [batch_size, feature_dim] (通常是2048维)
        """
        # 重置特征
        self.features = None
        
        # 前向传播
        _ = self.full_model(x)
        
        if self.features is None:
            raise RuntimeError("Feature extraction failed - hook did not capture features")
        
        return self.features


class I3DFeatureExtractorFlow(nn.Module):
    """I3D Flow模态特征提取器（暂未实现，保留接口）"""
    
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        # TODO: 实现Flow模态的I3D特征提取器
        raise NotImplementedError("Flow modality I3D feature extractor is not implemented yet")
    
    def forward(self, x):
        raise NotImplementedError("Flow modality I3D feature extractor is not implemented yet")



class MViT32FeatureExtractor(nn.Module):
    """MViT特征提取器，提取分类层前的特征"""

    feature_dim = 768  # MViT特征维度（没被用到。。

    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("PytorchVideo is required for MViT model")

        # 检查MViT模型可用性
        try:
            from pytorchvideo.models.hub import mvit_base_32x3
            self.full_model = mvit_base_32x3(pretrained=pretrained)
        except ImportError:
            # 如果MViT不可用，尝试其他方式
            logger.warning("MViT model not available in pytorchvideo, using alternative approach")
            raise ImportError("MViT model is not available")

        # 如果提供了自定义权重，加载它们
        if weights_path:
            self._load_custom_weights(weights_path)

        self.features = None
        self._register_hooks()

        logger.info(f"MViT feature extractor initialized (pretrained={pretrained})")

    def _load_custom_weights(self, weights_path: str):
        """加载自定义权重"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.full_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom weights from {weights_path}: {e}")

    def _register_hooks(self):
        """注册hooks来提取分类层前的特征"""

        def extract_features(module, input, output):
            # 保存768维特征，在分类层之前
            self.features = output.clone()

        # 在head.dropout层注册hook，提取768维特征
        for name, module in self.full_model.named_modules():
            if name == 'head.sequence_pool':  #
                module.register_forward_hook(extract_features)
                logger.debug(f"Registered hook at {name} for 768-dim feature extraction")
                break

    def forward(self, x, debug=False):
        """
        前向传播提取特征

        Args:
            x: 输入视频张量 [batch_size, channels, frames, height, width]
            debug: 是否打印调试信息

        Returns:
            特征张量 [batch_size, feature_dim]
        """
        if debug:
            print(f"MViT输入张量形状: {x.shape}")
            print(f"MViT输入张量设备: {x.device}")
            print(f"MViT输入张量数据类型: {x.dtype}")
            print(f"MViT输入张量值范围: [{x.min():.4f}, {x.max():.4f}]")

        # 重置特征
        self.features = None

        # 前向传播
        try:
            _ = self.full_model(x)
        except Exception as e:
            if debug:
                print(f"MViT前向传播错误: {e}")
                print(f"错误发生时输入形状: {x.shape}")
            raise

        if self.features is None:
            raise RuntimeError("Feature extraction failed - hook did not capture features")

        if debug:
            print(f"MViT输出特征形状: {self.features.shape}")
            print(f"MViT输出特征值范围: [{self.features.min():.4f}, {self.features.max():.4f}]")

        return self.features

    def print_model_structure(self, input_shape=(1, 3, 32, 224, 224)):
        """
        打印模型结构，显示每个层的名称和输出尺寸

        Args:
            input_shape: 输入张量的形状 (batch_size, channels, frames, height, width)
        """
        print(f"MViT模型结构分析 - 输入形状: {input_shape}")
        print("=" * 80)

        # 创建测试输入
        dummy_input = torch.randn(*input_shape)
        if next(self.full_model.parameters()).device != torch.device('cpu'):
            dummy_input = dummy_input.to(next(self.full_model.parameters()).device)

        # 注册钩子来捕获每层的输出
        layer_outputs = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = output.shape
                elif isinstance(output, (list, tuple)):
                    layer_outputs[name] = [o.shape if isinstance(o, torch.Tensor) else str(type(o)) for o in output]
                else:
                    layer_outputs[name] = str(type(output))

            return hook

        # 为所有命名模块注册钩子
        for name, module in self.full_model.named_modules():
            if name:  # 跳过根模块
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            try:
                output = self.full_model(dummy_input)
                print(f"最终输出形状: {output.shape}")
            except Exception as e:
                print(f"前向传播出错: {e}")

        # 打印每层信息
        print("\n层级结构详情:")
        print("-" * 80)
        for name, shape in layer_outputs.items():
            print(f"{name:50} -> {shape}")

        # 清理钩子
        for hook in hooks:
            hook.remove()

        print("=" * 80)

    def analyze_model_architecture(self):
        """分析模型架构的详细信息"""
        print("MViT模型架构分析:")
        print("=" * 60)

        total_params = sum(p.numel() for p in self.full_model.parameters())
        trainable_params = sum(p.numel() for p in self.full_model.parameters() if p.requires_grad)

        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型大小(MB): {total_params * 4 / 1024 / 1024:.2f}")

        print("\n主要模块结构:")
        print("-" * 40)
        for name, module in self.full_model.named_children():
            print(f"{name}: {type(module).__name__}")
            if hasattr(module, '__len__'):
                try:
                    print(f"  - 包含 {len(module)} 个子模块")
                except:
                    pass

        print("=" * 60)


class SlowFastFeatureExtractor(nn.Module):
    """SlowFast R101 16x8特征提取器，提取分类层前的特征"""

    feature_dim = 2304  # SlowFast特征维度（2048 + 256）

    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("PytorchVideo is required for SlowFast model")

        # 加载SlowFast R101 16x8
        try:
            from pytorchvideo.models.hub import slowfast_r101
            self.full_model = slowfast_r101(pretrained=pretrained)
        except ImportError:
            logger.error("SlowFast model not available in pytorchvideo")
            raise ImportError("SlowFast model is not available")

        # 可选自定义权重
        if weights_path:
            self._load_custom_weights(weights_path)

        self.features = None
        self._register_hooks()

        logger.info(f"SlowFast R101 16x8 feature extractor initialized (pretrained={pretrained})")

    def _load_custom_weights(self, weights_path: str):
        """加载自定义权重"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.full_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom weights from {weights_path}: {e}")

    def _register_hooks(self):
        """注册hooks来提取分类层前的特征（在blocks.5处展平为[N, 2304]）"""

        def feature_hook(module, inputs, output):
            # blocks.5 输出为 [N, 2304, 1, 1, 1]，需要展平为 [N, 2304]
            if isinstance(output, torch.Tensor):
                self.features = output.flatten(1)  # 从维度1开始展平
            else:
                raise RuntimeError(f"Unexpected hook output type at blocks.5: {type(output)}")

        hook_registered = False

        # 在 blocks.5 上注册 hook
        for name, module in self.full_model.named_modules():
            if name == 'blocks.5':
                module.register_forward_hook(feature_hook)
                logger.info(f"Registered hook at SlowFast feature extraction layer: {name}")
                hook_registered = True
                break

        # 如果 blocks.5 没找到，尝试其他可能的位置
        if not hook_registered:
            candidate_layers = ['blocks.6', 'head.pool', 'blocks.4']
            for layer_name in candidate_layers:
                for name, module in self.full_model.named_modules():
                    if name == layer_name:
                        module.register_forward_hook(feature_hook)
                        logger.info(f"Registered hook at SlowFast layer (fallback): {name}")
                        hook_registered = True
                        break
                if hook_registered:
                    break

        if not hook_registered:
            raise RuntimeError("Failed to register hook for SlowFast feature extraction layer")

    def create_slowfast_input(self, frames: torch.Tensor, alpha: int = 4):
        """
        将单一路输入切分成 SlowFast 输入
        Args:
            frames: (B, C, T, H, W)
            alpha: 时间分辨率比例 (默认 4)
        Returns:
            [slow_path, fast_path]
        """
        fast_pathway = frames  # fast 原始输入
        # 修复：确保索引在与输入tensor相同的设备上
        device = frames.device
        index = torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // alpha, device=device).long()
        slow_pathway = torch.index_select(frames, 2, index)
        return [slow_pathway, fast_pathway]

    def forward(self, x, debug: bool = False):
        """
        前向传播提取特征
        Args:
            x: [batch, 3, T, H, W]
            debug: 是否打印调试信息
        Returns:
            [batch, feature_dim] = [N, 2304]
        """
        if debug:
            print(f"SlowFast原始输入张量形状: {x.shape}")
            print(f"SlowFast输入张量设备: {x.device}")
            print(f"SlowFast输入张量数据类型: {x.dtype}")
            min_val = float(x.min().detach().cpu())
            max_val = float(x.max().detach().cpu())
            print(f"SlowFast输入张量值范围: [{min_val:.4f}, {max_val:.4f}]")

        # 重置特征
        self.features = None

        try:
            # 使用 create_slowfast_input 方法来准备输入
            inputs = self.create_slowfast_input(x, alpha=4)

            # 确保输入tensor在正确的设备上
            model_device = next(self.full_model.parameters()).device
            if x.device != model_device:
                x = x.to(model_device)

                print(f"输入tensor已移动到设备: {x.device}")
            if debug:
                print("SlowFast准备后的输入：")
                for i, p in enumerate(inputs):
                    print(f"  Pathway {i} 形状: {tuple(p.shape)}")

            # 前向传播（特征由 hook 在 blocks.5 处捕获）
            _ = self.full_model(inputs)

        except Exception as e:
            if debug:
                print(f"SlowFast前向传播错误: {e}")
                print(f"错误发生时输入形状: {x.shape}")
                import traceback as _tb
                _tb.print_exc()
            raise

        if self.features is None:
            raise RuntimeError("Feature extraction failed - hook did not capture features")

        if debug:
            print(f"SlowFast输出特征形状: {tuple(self.features.shape)}")
            print(f"SlowFast输出特征值范围: [{self.features.min():.4f}, {self.features.max():.4f}]")

        return self.features

    def print_model_structure(self, input_shape=(1, 3, 32, 224, 224)):
        """打印模型结构，显示每个层的名称和输出尺寸"""
        print(f"SlowFast模型结构分析 - 输入形状: {input_shape}")
        print("=" * 80)

        dummy_input = torch.randn(*input_shape)
        if next(self.full_model.parameters()).is_cuda:
            dummy_input = dummy_input.to(next(self.full_model.parameters()).device)

        # 按 SlowFast 要求打包输入
        model_input = self.create_slowfast_input(dummy_input, alpha=4)

        layer_outputs = {}
        hooks = []

        def make_hook(name):
            def hook(module, _in, out):
                if isinstance(out, torch.Tensor):
                    layer_outputs[name] = str(tuple(out.shape))
                elif isinstance(out, (list, tuple)):
                    shapes = [str(tuple(o.shape)) if isinstance(o, torch.Tensor) else str(type(o)) for o in out]
                    layer_outputs[name] = f"[{', '.join(shapes)}]"
                else:
                    layer_outputs[name] = str(type(out))
            return hook

        # 只注册重要的层，避免输出过多
        important_layers = []
        for name, module in self.full_model.named_modules():
            if name and any(k in name.lower() for k in ['block', 'head', 'pool']):
                if len(name.split('.')) <= 2:  # 只看较高层级的模块
                    important_layers.append((name, module))

        for name, module in important_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            try:
                out = self.full_model(model_input)
                if isinstance(out, torch.Tensor):
                    print(f"最终输出形状: {tuple(out.shape)}")
                else:
                    print(f"最终输出类型: {type(out)}")
            except Exception as e:
                print(f"前向传播出错: {e}")

        print("\n重要层级结构:")
        print("-" * 80)
        for name in sorted(layer_outputs.keys()):
            print(f"{name:40} -> {layer_outputs[name]}")

        for h in hooks:
            h.remove()

        print("=" * 80)

    def analyze_model_architecture(self):
        """分析模型架构的详细信息"""
        print("SlowFast模型架构分析:")
        print("=" * 60)

        total_params = sum(p.numel() for p in self.full_model.parameters())
        trainable_params = sum(p.numel() for p in self.full_model.parameters() if p.requires_grad)

        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型大小(MB): {total_params * 4 / 1024 / 1024:.2f}")

        print("\n主要模块结构:")
        print("-" * 40)
        for name, module in self.full_model.named_children():
            print(f"{name}: {type(module).__name__}")
            if hasattr(module, '__len__'):
                try:
                    print(f"  - 包含 {len(module)} 个子模块")
                except Exception:
                    pass

        print("=" * 60)




if __name__ == "__main__":
    print("SlowFast R101 16x8模型分析")
    print("=" * 100)

    # 创建SlowFast模型实例
    slowfast_extractor = SlowFastFeatureExtractor(pretrained=True)

    # 打印详细的模型结构
    slowfast_extractor.print_model_structure()

    # 分析模型架构
    slowfast_extractor.analyze_model_architecture()

    # 在前向传播时启用调试
    dummy_input_slowfast = torch.randn(1, 3, 32, 224, 224)  # SlowFast输入 (16帧)
    features_slowfast = slowfast_extractor(dummy_input_slowfast, debug=True)


    # slowfast_extractor = SlowFastFeatureExtractor(pretrained=True)
    # slowfast_extractor.debug_print_all_layers(input_shape=(1, 3, 32, 224, 224))

    print("=" * 100)
    print("MViT32模型分析")
    print("=" * 100)

    # 创建MViT32模型实例
    mvit32_extractor = MViT32FeatureExtractor(pretrained=True)

    # 打印详细的模型结构
    mvit32_extractor.print_model_structure()

    # 分析模型架构
    mvit32_extractor.analyze_model_architecture()

    # 在前向传播时启用调试
    dummy_input_mvit32 = torch.randn(1, 3, 32, 224, 224)  # MViT32输入
    features_mvit32 = mvit32_extractor(dummy_input_mvit32, debug=True)

    print("=" * 100)


    print("\n" + "=" * 100)
    print("I3D模型分析")
    print("=" * 100)

    # 创建I3D模型实例
    i3d_extractor = I3DFeatureExtractorRGB(pretrained=True)

    # 打印详细的模型结构
    i3d_extractor.print_model_structure()

    # 分析模型架构
    i3d_extractor.analyze_model_architecture()

    # 在前向传播时启用调试
    dummy_input_i3d = torch.randn(1, 3, 8, 224, 224)  # I3D输入
    features_i3d = i3d_extractor(dummy_input_i3d, debug=True)

    print("\n" + "=" * 100)
    print("模型对比")
    print("=" * 100)
    print(f"MViT32特征维度: {mvit32_extractor.feature_dim}")

    print(f"I3D输入帧数: 16")

    # 特征形状对比
    print(f"\n实际特征输出形状:")
    print(f"MViT32: {features_mvit32.shape}")

    # 显存使用对比（如果在GPU上）
    if torch.cuda.is_available():
        print(f"\n显存使用情况:")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")