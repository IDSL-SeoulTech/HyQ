""" MobileViT

Paper:
V1: `MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer` - https://arxiv.org/abs/2110.02178
V2: `Separable Self-attention for Mobile Vision Transformers` - https://arxiv.org/abs/2206.02680

MobileVitBlock and checkpoints adapted from https://github.com/apple/ml-cvnets (original copyright below)
License: https://github.com/apple/ml-cvnets/blob/main/LICENSE (Apple open source)

Rest of code, ByobNet, and Transformer block hacked together by / Copyright 2022, Ross Wightman
"""
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import math
from typing import Callable, Tuple, Optional
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

# from timm.layers import to_2tuple, make_divisible, GroupNorm1, ConvMlp, DropPath, is_exportable
from .layers import to_2tuple, make_divisible, GroupNorm1, ConvMlp, DropPath, is_exportable
from ._builder import build_model_with_cfg
from ._features_fx import register_notrace_module
from ._registry import register_model, generate_default_cfgs, register_model_deprecations
from .byobnet import register_block, ByoBlockCfg, ByoModelCfg, ByobNet, LayerFn, num_groups
# from timm.models.vision_transformer import Block as TransformerBlock
from .vit_quant import Block as TransformerBlock
from .layers_quant import DropPath, HybridEmbed, Mlp, PatchEmbed, trunc_normal_
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear, QOlc

__all__ = ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']


def _inverted_residual_block(d, c, s, br=4.0):
    # inverted residual is a bottleneck block with bottle_ratio > 1 applied to in_chs, linear output, gs=1 (depthwise)
    return ByoBlockCfg(
        type='bottle', d=d, c=c, s=s, gs=1, br=br,
        block_kwargs=dict(bottle_in=True, linear_out=True))


def _mobilevit_block(d, c, s, transformer_dim, transformer_depth, patch_size=4, br=4.0):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type='mobilevit', d=1, c=c, s=1,
            block_kwargs=dict(
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                patch_size=patch_size,
            )
        )
    )



model_cfgs = dict(
    mobilevit_xxs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=16, s=1, br=2.0),
            _inverted_residual_block(d=3, c=24, s=2, br=2.0),
            _mobilevit_block(d=1, c=48, s=2, transformer_dim=64, transformer_depth=2, patch_size=2, br=2.0),
            _mobilevit_block(d=1, c=64, s=2, transformer_dim=80, transformer_depth=4, patch_size=2, br=2.0),
            _mobilevit_block(d=1, c=80, s=2, transformer_dim=96, transformer_depth=3, patch_size=2, br=2.0),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=320,
    ),

    mobilevit_xs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=48, s=2),
            _mobilevit_block(d=1, c=64, s=2, transformer_dim=96, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=80, s=2, transformer_dim=120, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=384,
    ),

    mobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=640,
    ),

    semobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=1/8),
        num_features=640,
    ),
)


@register_notrace_module
class MobileVitBlock(nn.Module):
    """ MobileViT block
        Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            dilation: Tuple[int, int] = (1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim: Optional[int] = None,
            transformer_depth: int = 2,
            patch_size: int = 8,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: int = 0.,
            no_fusion: bool = False,
            drop_path_rate: float = 0.,
            layers: LayerFn = None,
            transformer_norm_layer: Callable = partial(QIntLayerNorm, eps=1e-6),
            FQcfg = None,
            **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        ## NJ
        self.qact_shortcut = QAct(
                    bit_type=FQcfg.BIT_TYPE_A,
                    calibration_mode=FQcfg.CALIBRATION_MODE_A,
                    observer_str=FQcfg.OBSERVER_A,
                    quantizer_str=FQcfg.QUANTIZER_A)
        
        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=stride, groups=groups, dilation=dilation[0])
        self.conv_kxk.conv = QConv2d(in_chs,
                    in_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    bias=False,
                    padding=1, 
                    dilation=dilation[0],
                    bit_type=FQcfg.BIT_TYPE_W,
                    calibration_mode=FQcfg.CALIBRATION_MODE_W,
                    observer_str=FQcfg.OBSERVER_W,
                    quantizer_str=FQcfg.QUANTIZER_W)
        self.qact_kxk = QAct(
                          bit_type=FQcfg.BIT_TYPE_A,
                          calibration_mode=FQcfg.CALIBRATION_MODE_A,
                          observer_str=FQcfg.OBSERVER_A,
                          quantizer_str=FQcfg.QUANTIZER_A)
        
        self.conv_1x1 = QConv2d(in_chs,
                            transformer_dim,
                            kernel_size=1,
                            bias=False,
                            bit_type=FQcfg.BIT_TYPE_W,
                            calibration_mode=FQcfg.CALIBRATION_MODE_W,
                            observer_str=FQcfg.OBSERVER_W,
                            quantizer_str=FQcfg.QUANTIZER_W)
        self.qact_1x1 = QAct(
                          bit_type=FQcfg.BIT_TYPE_A,
                          calibration_mode=FQcfg.CALIBRATION_MODE_A,
                          observer_str=FQcfg.OBSERVER_A,
                          quantizer_str=FQcfg.QUANTIZER_A)

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path_rate,
                act_layer=layers.act,
                norm_layer=transformer_norm_layer,     
                cfg=FQcfg,           
            )
            for _ in range(transformer_depth)
        ])
        
        self.norm = transformer_norm_layer(transformer_dim)
        self.qact_norm = QAct(
                        bit_type=FQcfg.BIT_TYPE_A,
                        calibration_mode=FQcfg.CALIBRATION_MODE_A,
                        observer_str=FQcfg.OBSERVER_A,
                        quantizer_str=FQcfg.QUANTIZER_A)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1)
        self.conv_proj.conv = QConv2d(transformer_dim,
            out_chs,
            kernel_size=1,
            stride=1,
            bias=False,
            bit_type=FQcfg.BIT_TYPE_W,
            calibration_mode=FQcfg.CALIBRATION_MODE_W,
            observer_str=FQcfg.OBSERVER_W,
            quantizer_str=FQcfg.QUANTIZER_W)
        self.qact_proj = QAct(
                          bit_type=FQcfg.BIT_TYPE_A,
                          calibration_mode=FQcfg.CALIBRATION_MODE_A,
                          observer_str=FQcfg.OBSERVER_A,
                          quantizer_str=FQcfg.QUANTIZER_A)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = layers.conv_norm_act(in_chs + out_chs, out_chs, kernel_size=kernel_size, stride=1)
            self.conv_fusion.conv = QConv2d(in_chs + out_chs,
                out_chs,
                kernel_size=kernel_size,
                padding=1,
                stride=1,
                bias=False,
                bit_type=FQcfg.BIT_TYPE_W,
                calibration_mode=FQcfg.CALIBRATION_MODE_W,
                observer_str=FQcfg.OBSERVER_W,
                quantizer_str=FQcfg.QUANTIZER_W)
            self.qact_fusion = QAct(
                    bit_type=FQcfg.BIT_TYPE_A,
                    calibration_mode=FQcfg.CALIBRATION_MODE_A,
                    observer_str=FQcfg.OBSERVER_A,
                    quantizer_str=FQcfg.QUANTIZER_A)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        shortcut = self.qact_shortcut(shortcut)

        # Local representation
        x = self.conv_kxk(x)
        #print(x.max(), x.min(), x.mean(), x.var())
        x = self.qact_kxk(x)
        x = self.conv_1x1(x)
        x = self.qact_1x1(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)

        #Global representations
        for i, blk in enumerate(self.transformer):
            last_quantizer = self.qact_1x1.quantizer if i ==0 else self.transformer[i - 1].qact4.quantizer
            x = blk(x, last_quantizer)
        #x = self.transformer(x)
        x = self.norm(x, self.transformer[-1].qact4.quantizer, self.qact_norm.quantizer)
        x = self.qact_norm(x)
        
        
        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)
        x = self.qact_proj(x)
        
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
            x = self.qact_fusion(x)

        return x



register_block('mobilevit', MobileVitBlock)


def _create_mobilevit(variant, cfg_variant=None, pretrained=False, FQcfg=None, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        FQcfg= FQcfg,
        **kwargs)



def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': (0., 0., 0.), 'std': (1., 1., 1.),
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'fixed_input_size': False,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mobilevit_xxs.cvnets_in1k': _cfg(hf_hub_id='timm/'),
    'mobilevit_xs.cvnets_in1k': _cfg(hf_hub_id='timm/'),
    'mobilevit_s.cvnets_in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def mobilevit_xxs(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit('mobilevit_xxs', pretrained=pretrained, **kwargs)


@register_model
def mobilevit_xs(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit('mobilevit_xs', pretrained=pretrained, **kwargs)


@register_model
def mobilevit_s(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit('mobilevit_s', pretrained=pretrained, **kwargs)