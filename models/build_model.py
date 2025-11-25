# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
模型构建模块 - 整合SPAI和Effort
"""

import sys
from pathlib import Path

# 添加fusion路径到sys.path，使其可以独立运行
fusion_dir = Path(__file__).parent.parent
if str(fusion_dir) not in sys.path:
    sys.path.insert(0, str(fusion_dir))
# 如果需要在父目录查找，也添加父目录
parent_dir = fusion_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from typing import Optional, Union
import torch
from torch import nn
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models import filters
from models.svd_backbone import CLIPBackboneWithSVD, CLIP_MEAN, CLIP_STD
from models.sid import (
    FrequencyRestorationEstimator,
    ClassificationHead,
    PatchBasedMFViT,
    MFViT,
    _init_weights
)


def build_fusion_model(
    config,
    svd_rank: int = 768,
    use_svd: bool = True,
    clip_model: str = "ViT-B/16"
):
    """
    构建融合模型（SPAI + Effort SVD）
    
    Args:
        config: 配置对象
        svd_rank: SVD保留的rank
        use_svd: 是否使用SVD
        clip_model: CLIP模型名称
    
    Returns:
        构建好的模型
    """
    # 构建带SVD的CLIP backbone
    # 注意：CLIP ViT-B/16的embed_dim是768，ViT-L/14是1024
    embed_dim_map = {
        "ViT-B/16": 768,
        "ViT-L/14": 1024
    }
    embed_dim = embed_dim_map.get(clip_model, 768)
    
    # 如果config中的embed_dim与CLIP不匹配，需要调整
    if hasattr(config, 'MODEL') and hasattr(config.MODEL, 'VIT'):
        if config.MODEL.VIT.EMBED_DIM != embed_dim:
            print(f"Warning: Config embed_dim ({config.MODEL.VIT.EMBED_DIM}) != CLIP embed_dim ({embed_dim})")
            print(f"Using CLIP embed_dim: {embed_dim}")
    
    vit = CLIPBackboneWithSVD(
        clip_model=clip_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        svd_rank=svd_rank,
        use_svd=use_svd,
        apply_svd_on_init=False  # 推迟SVD到加载完SPAI权重之后
    )
    
    # 获取CLIP的实际embed_dim
    embed_dim_map = {
        "ViT-B/16": 768,
        "ViT-L/14": 1024
    }
    actual_embed_dim = embed_dim_map.get(clip_model, 768)
    
    # 构建特征处理器
    # 注意：CLIP的中间层数量是固定的（ViT-B/16有12层，ViT-L/14有24层）
    # 这里使用config中的INTERMEDIATE_LAYERS，但需要确保数量正确
    intermediate_layers = config.MODEL.VIT.INTERMEDIATE_LAYERS
    if clip_model == "ViT-B/16" and len(intermediate_layers) > 12:
        intermediate_layers = list(range(12))
        print(f"Warning: CLIP ViT-B/16 only has 12 layers, using {intermediate_layers}")
    elif clip_model == "ViT-L/14" and len(intermediate_layers) > 24:
        intermediate_layers = list(range(24))
        print(f"Warning: CLIP ViT-L/14 only has 24 layers, using {intermediate_layers}")
    
    fre = FrequencyRestorationEstimator(
        features_num=len(intermediate_layers),
        input_dim=actual_embed_dim,
        proj_dim=config.MODEL.VIT.PROJECTION_DIM,
        proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
        patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
        patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
        proj_last_layer_activation_type=config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE,
        original_image_features_branch=config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH,
        dropout=config.MODEL.SID_DROPOUT,
        disable_reconstruction_similarity=config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY
    )
    
    # 计算分类向量维度
    cls_vector_dim: int = 6 * len(intermediate_layers)
    if (config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH
            and config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY):
        cls_vector_dim = config.MODEL.VIT.PROJECTION_DIM
    elif config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH:
        cls_vector_dim += config.MODEL.VIT.PROJECTION_DIM
    
    # 构建分类头
    cls_head = ClassificationHead(
        input_dim=cls_vector_dim,
        num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
        mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
        dropout=config.MODEL.SID_DROPOUT
    )
    
    # 根据分辨率模式选择模型
    if config.MODEL.RESOLUTION_MODE == "fixed":
        model = MFViT(
            vit=vit,
            features_processor=fre,
            cls_head=cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_size=config.DATA.IMG_SIZE,
            frozen_backbone=False,  # SVD模式下backbone部分可训练
            initialization_scope="local"  # 只初始化新添加的组件
        )
    elif config.MODEL.RESOLUTION_MODE == "arbitrary":
        model = PatchBasedMFViT(
            vit=vit,
            features_processor=fre,
            cls_head=cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_patch_size=config.DATA.IMG_SIZE,
            img_patch_stride=config.MODEL.PATCH_VIT.PATCH_STRIDE if hasattr(config.MODEL, 'PATCH_VIT') else 224,
            cls_vector_dim=cls_vector_dim,
            attn_embed_dim=config.MODEL.PATCH_VIT.ATTN_EMBED_DIM if hasattr(config.MODEL, 'PATCH_VIT') else 1536,
            num_heads=config.MODEL.PATCH_VIT.NUM_HEADS if hasattr(config.MODEL, 'PATCH_VIT') else 12,
            dropout=config.MODEL.SID_DROPOUT,
            minimum_patches=config.MODEL.PATCH_VIT.MINIMUM_PATCHES if hasattr(config.MODEL, 'PATCH_VIT') else 1,
            frozen_backbone=True,
            initialization_scope="local"
        )
    else:
        raise RuntimeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")
    
    return model


def apply_svd_to_model_backbone(
    model: nn.Module,
    svd_rank: int,
    device: str | torch.device = "cuda"
) -> None:
    """在加载完SPAI权重后，再对backbone执行SVD压缩。"""
    backbone = None
    if hasattr(model, "vit"):
        backbone = model.vit
    elif hasattr(model, "mfvit") and hasattr(model.mfvit, "vit"):
        backbone = model.mfvit.vit

    if backbone is None:
        raise RuntimeError("Unable to locate backbone for SVD application.")

    if isinstance(backbone, CLIPBackboneWithSVD):
        backbone.apply_svd(svd_rank, str(device))
    else:
        raise TypeError("Backbone does not support deferred SVD application.")


def load_pretrained_weights(
    model,
    spai_checkpoint: Optional[str] = None,
    effort_checkpoint: Optional[str] = None,
    strict: bool = False
):
    """
    加载预训练权重
    
    Args:
        model: 模型实例
        spai_checkpoint: SPAI预训练权重路径
        effort_checkpoint: Effort预训练权重路径
        strict: 是否严格匹配
    """
    # 先加载Effort权重到backbone
    if effort_checkpoint is not None:
        if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'vit'):
            model.mfvit.vit.load_pretrained_effort(effort_checkpoint)
        elif hasattr(model, 'vit'):
            model.vit.load_pretrained_effort(effort_checkpoint)
    
    # 再加载SPAI权重（会覆盖backbone，但保留SVD结构）
    if spai_checkpoint is not None:
        print(f"Loading SPAI pretrained weights from {spai_checkpoint}")
        checkpoint = torch.load(spai_checkpoint, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 处理键名映射
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 移除可能的prefix
            if k.startswith('mfvit.'):
                k = k[6:]
            if k.startswith('vit.'):
                # 对于backbone，只加载非SVD参数
                if any(x in k for x in ['S_residual', 'U_residual', 'V_residual']):
                    continue  # 跳过SVD残差参数，保持Effort的初始化
                k = k[4:]
            
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(pretrained_dict)} parameters from SPAI checkpoint")

