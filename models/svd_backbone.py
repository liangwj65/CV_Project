# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SVD-based CLIP Backbone for SPAI
结合Effort的SVD方法和SPAI的CLIPBackbone
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


CLIP_MEAN: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class SVDResidualLinear(nn.Module):
    """SVD分解的残差线性层，来自Effort方法"""
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None, device='cuda'):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # 保留的前r个奇异值（主权重）

        # 主权重（固定，不训练）
        # 正确写法 1（推荐）：直接用 zeros + copy，自动在正确设备
        self.weight_main = nn.Parameter(
            torch.zeros(out_features, in_features, device=device, dtype=torch.float16),
            requires_grad=False
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)
        
        # SVD残差组件（可训练）
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None

    @property
    def weight(self) -> torch.Tensor:
        """提供仿nn.Linear接口，兼容需要直接访问weight的模块"""
        device = self.weight_main.device  # 关键：以 weight_main 为准
        dtype = self.weight_main.dtype  # 原始 dtype（通常是 half）

        if self.S_residual is not None:
            # 强制把残差三个矩阵搬到正确设备和类型（防炸核心！）
            U = self.U_residual.to(device=device, dtype=dtype)
            S = self.S_residual.to(device=device, dtype=dtype)
            V = self.V_residual.to(device=device, dtype=dtype)
            if not torch.isfinite(S).all():
                print(f"    S_residual contains:")
                print(f"      inf  = {torch.isinf(S).any().item()}")
                print(f"      -inf = {(S == -float('inf')).any().item()}")
                print(f"      nan  = {torch.isnan(S).any().item()}")
                print(f"      max  = {S.max().item():.2e}")
                print(f"      min  = {S.min().item():.2e}")
                print(f"      first 10 values: {S.flatten()[:10].cpu().tolist()}")
                print(f"    Layer: {self}\n")
            # 高效重构：避免 torch.diag 分配大矩阵
            residual_weight = torch.einsum('ij,j,jk->ik', U, S, V)  # 比 diag 快 3-5 倍
            # residual_weight = (U @ torch.diag_embed(S) @ V)  # 备选，老写法

            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main

        # ====== 下面这段是 autocast 兼容，保留但精简 ======
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()  # 2024+ 推荐写法，自动支持 bf16
            if weight.dtype != target_dtype:
                weight = weight.to(dtype=target_dtype)

        return weight


    def forward(self, x):
        weight = self.weight
        bias = self.bias

        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)

        return F.linear(x, weight, bias)


# 只改这几个地方！！！

def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        device = module.weight.device  # ← 关键：记住原始设备
        dtype = module.weight.dtype  # ← 记住原始 dtype（通常 half）

        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # 创建新模块
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias)

        # 把原始权重搬过来（已经在正确设备）
        new_module.weight_main.data.copy_(module.weight.data)
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # SVD 分解（强制在正确设备 + float32 计算）
        weight_4svd = module.weight.data.float()  # 转 fp32 算 SVD
        U, S, Vh = torch.linalg.svd(weight_4svd, full_matrices=False)

        r = min(r, len(S))

        # 主成分（固定）
        weight_main = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).to(dtype).to(device)
        new_module.weight_main.data.copy_(weight_main)

        # 残差成分（可训练）
        if len(S) > r:
            U_res = U[:, r:].to(dtype).to(device)
            #S_res = S[r:].to(dtype).to(device)
            Vh_res = Vh[r:, :].to(dtype).to(device)

            # 关键三连：残差奇异值必须大幅缩放 + clamp 防止负值
            S_res = S[r:].to(dtype).to(device)
            # 1. 缩放到非常小（原始残差奇异值通常是噪声，直接训练会爆炸）
            # S_res = S_res_raw / S_res_raw.max() * 1e-4  # 推荐 1e-4 ~ 1e-3
            # # 2. 强制 clamp 防止变成负数（奇异值必须 ≥0）
            # S_res = S_res.clamp(min=1e-8,max=1e-3)
            # S_res = S_res.clamp(min=1e-8, max=1e-3)

            new_module.U_residual = nn.Parameter(U_res.contiguous())
            new_module.S_residual = nn.Parameter(S_res.contiguous())
            new_module.V_residual = nn.Parameter(Vh_res.contiguous())
        else:
            new_module.U_residual = new_module.S_residual = new_module.V_residual = None

        return new_module
    return module


def apply_svd_to_clip_self_attn(model, r, target_dtype: torch.dtype | None = None):
    """对CLIP模型的self_attn层应用SVD分解"""
    for name, module in model.named_children():
        if 'self_attn' in name or 'attn' in name:
            # 替换self_attn中的nn.Linear层
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # 获取父模块
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # 替换为SVD版本
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
                    parent_module.to('cuda')
        else:
            # 递归应用到子模块
            apply_svd_to_clip_self_attn(module, r)
    
    # 设置参数的可训练性
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if target_dtype is not None:
        cast_svd_linear_modules_dtype(model, target_dtype)
    
    return model


def cast_svd_linear_modules_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Recursively cast all SVDResidualLinear params to the desired dtype."""
    for child in module.modules():
        if isinstance(child, SVDResidualLinear):
            child.weight_main.data = child.weight_main.data.to(dtype)
            if child.bias is not None:
                child.bias.data = child.bias.data.to(dtype)
            if isinstance(child.S_residual, nn.Parameter):
                child.S_residual.data = child.S_residual.data.to(dtype)
            if isinstance(child.U_residual, nn.Parameter):
                child.U_residual.data = child.U_residual.data.to(dtype)
            if isinstance(child.V_residual, nn.Parameter):
                child.V_residual.data = child.V_residual.data.to(dtype)


class CLIPBackboneWithSVD(nn.Module):
    """带SVD分解的CLIP Backbone，用于SPAI"""
    def __init__(
        self,
        clip_model: str = "ViT-B/16",
        device: str = "cpu",
        svd_rank: int = 768,  # SVD保留的rank，默认保留大部分（ViT-B/16的embed_dim是768）
        use_svd: bool = True,
        apply_svd_on_init: bool = True
    ) -> None:
        super().__init__()
        self.use_svd = use_svd
        self.clip_model = clip_model
        self._svd_rank = svd_rank
        self._svd_applied = False

        # 加载CLIP模型
        self.clip, self.preprocess = clip.load(clip_model, device=device)
        
        if self.use_svd and apply_svd_on_init:
            self._apply_svd_internal(svd_rank, device)
        elif not self.use_svd:
            for _, param in self.clip.named_parameters():
                param.requires_grad = False

        # 注册hooks以获取中间层输出
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理图像批次并返回中间层特征"""
        # 确保LayerNorm参数始终为FP32（CLIP的要求）
        if self.clip.visual.transformer.resblocks[1].ln_1.weight.dtype != torch.float32:
            for m in self.clip.modules():
                if isinstance(m, clip.model.LayerNorm):
                    m.float()

        self.clip.encode_image(x)
        # 获取所有hook的输出并堆叠
        hook_outputs = [h.output for h in self.hooks]
        if len(hook_outputs) > 0:
            x = torch.stack(hook_outputs, dim=2)[1:, :, :, :]  # 跳过第一个（可能是CLS token）
            x = torch.permute(x, (1, 2, 0, 3))  # B x N x L x D
        else:
            # 如果没有hooks，返回pooled output
            x = self.clip.visual(x)
            x = x.unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x D

        return x

    def _apply_svd_internal(self, svd_rank: int, device: str) -> None:
        if self._svd_applied or not self.use_svd:
            return

        print(f"Applying SVD to CLIP backbone with rank={svd_rank}")
        target_dtype = None
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            target_dtype = torch.float16
        self.clip.visual = apply_svd_to_clip_self_attn(self.clip.visual, r=svd_rank, target_dtype=target_dtype)

        num_param = sum(p.numel() for p in self.clip.visual.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.clip.visual.parameters())
        print(f'CLIP Backbone - Total parameters: {num_total_param:,}, Trainable parameters: {num_param:,} ({num_param/num_total_param*100:.2f}%)')
        self._svd_applied = True

    def apply_svd(self, svd_rank: int | None = None, device: str = "cpu") -> None:
        if not self.use_svd:
            return
        rank = svd_rank or self._svd_rank
        self._apply_svd_internal(rank, device)

    def load_pretrained_effort(self, checkpoint_path: str):
        """加载Effort预训练的权重"""
        print(f"Loading Effort pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Effort的checkpoint可能包含'model'键
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 尝试加载权重（可能需要键名映射）
        model_dict = self.clip.visual.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 处理不同的键名格式
            if 'vision_model' in k:
                k = k.replace('vision_model.', '')
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
        
        model_dict.update(pretrained_dict)
        self.clip.visual.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} parameters from Effort checkpoint")

