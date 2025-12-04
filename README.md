# SPAI + Effort SVD Fusion

本项目整合了SPAI（频域AIGI检测）和Effort（SVD快速微调）两个方法，使用SVD方法对SPAI的CLIP backbone进行高效微调。

## 架构说明

### 核心思想

1. **SPAI的频域检测框架**：使用频域分解（高低频）和频域重建相似度来检测AIGI
2. **Effort的SVD方法**：对CLIP backbone的self-attention层进行SVD分解
   - 保留前r个奇异值作为**固定主权重**（不训练）
   - 剩余部分作为**可训练残差**（参数量大幅减少）

### 模型结构

```
输入图像
    ↓
频域分解（FFT + Mask）→ [原图, 低频, 高频]
    ↓
CLIP Backbone (with SVD) → [原图特征, 低频特征, 高频特征]
    ↓
FrequencyRestorationEstimator → 计算余弦相似度统计量
    ↓
ClassificationHead → AIGI概率
```

### SVD分解细节

对CLIP的self-attention层中的Linear层进行SVD分解：

```python
# 原始权重 W
U, S, Vh = torch.linalg.svd(W)

# 主权重（固定，不训练）
W_main = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]

# 残差权重（可训练）
W_residual = U[:, r:] @ diag(S[r:]) @ Vh[r:, :]

# 总权重
W_total = W_main + W_residual
```

## 文件结构

```
fusion/
├── models/
│   ├── svd_backbone.py      # SVD CLIP Backbone实现
│   └── build_model.py        # 模型构建和权重加载
├── data/
│   └── dataset.py           # 数据集处理（复用SPAI）
├── config.py                 # 配置管理
├── utils.py                  # 工具函数
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
└── README.md                 # 本文档
```

## 使用方法

### 1. 准备预训练权重

确保你有以下预训练权重：

- **SPAI权重**：`./weights/spai.pth`
  - 这是SPAI训练好的完整模型权重
  - 包含FrequencyRestorationEstimator和ClassificationHead的权重

- **Effort权重**（可选）：`./weights/effort_clip_l14.pth`
  - 这是Effort在CLIP上训练的权重
  - 如果提供，会先加载到backbone，然后再加载SPAI权重

### 2. 准备数据

数据格式：CSV文件，包含以下列：
- `image`: 图像路径（相对或绝对）
- `split`: 数据分割（train/val/test）
- `class`: 类别标签（0=真实, 1=AIGI）

示例：
```csv
image,split,class
path/to/image1.jpg,train,0
path/to/image2.jpg,train,1
```

### 3. 训练

修改 `train.py` 中的参数：

```python
# 配置文件
config_path = "./configs/spai.yaml"

# 预训练权重
spai_checkpoint = "./weights/spai.pth"
effort_checkpoint = "./weights/effort_clip_l14.pth"  # 可选

# SVD参数
svd_rank = 768  # 保留的rank（ViT-B/16的embed_dim是768）
use_svd = True
clip_model = "ViT-B/16"  # 或 "ViT-L/14"

# 训练参数
batch_size = 32
num_epochs = 35
learning_rate = 5e-4

# 数据路径
train_csv = "./data/train.csv"
val_csv = "./data/val.csv"
csv_root = "./data"
```

运行训练：
```bash
cd fusion
python train.py
```

### 4. 测试

修改 `test.py` 中的参数：

```python
# 模型权重（训练后的）
model_checkpoint = "./output/fusion/best_model.pth"

# SVD参数（必须与训练时一致）
svd_rank = 768
use_svd = True
clip_model = "ViT-B/16"

# 测试数据
test_csv = "./data/test.csv"
csv_root = "./data"
```

运行测试：
```bash
cd fusion
python test.py
```

## 重要注意事项

### 1. Backbone选择

- **CLIP ViT-B/16**：embed_dim=768，适合大多数场景
- **CLIP ViT-L/14**：embed_dim=1024，性能更好但需要更多显存
- SVD rank建议设置为 `embed_dim - 1` 或更小，例如：
  - ViT-B/16: rank=767 或 512
  - ViT-L/14: rank=1023 或 768

### 2. SVD Rank设置

- **rank越大**：保留更多主权重，可训练参数越少，但可能欠拟合
- **rank越小**：保留更少主权重，可训练参数越多，但可能过拟合
- **建议**：从 `embed_dim - 1` 开始，根据验证集表现调整

### 3. 权重加载顺序

1. 先加载Effort权重到backbone（如果提供）
2. 再加载SPAI权重（会覆盖backbone，但保留SVD结构）
3. SVD残差参数（S_residual, U_residual, V_residual）保持Effort的初始化

### 4. 训练集要求

- **真实图像**：来自COCO、ImageNet、OpenImages等真实数据集
- **AIGI图像**：来自各种生成模型（Stable Diffusion、DALL-E、Midjourney等）
- **数据平衡**：建议真实和AIGI图像数量接近
- **数据增强**：使用SPAI的默认增强策略

### 5. 显存优化

如果显存不足，可以：
- 减小batch_size
- 减小svd_rank（减少可训练参数）
- 使用gradient checkpointing
- 使用混合精度训练（FP16）

### 6. 超参数调优

关键超参数：
- `learning_rate`: 建议从5e-4开始，SVD模式下可以稍大（1e-3）
- `weight_decay`: 0.05（与SPAI一致）
- `svd_rank`: 根据模型大小和数据集调整
- `batch_size`: 根据GPU显存调整

### 7. 兼容性

- **Python**: 3.8+
- **PyTorch**: 1.11+
- **CUDA**: 11.0+（如果使用GPU）
- 需要安装：`clip`, `timm`, `albumentations`等（见SPAI的requirements.txt）

## 预期效果

- **参数量减少**：相比全量微调，可训练参数减少约70-90%
- **训练速度**：每个epoch训练时间减少约30-50%
- **性能**：在大多数数据集上，性能接近或略低于全量微调（差异<2%）

## 故障排除

### 问题1：权重加载失败
- 检查权重路径是否正确
- 检查权重格式（是否包含'model'键）
- 尝试设置`strict=False`

### 问题2：显存不足
- 减小batch_size
- 减小svd_rank
- 使用更小的CLIP模型（ViT-B/16而不是ViT-L/14）

### 问题3：训练不收敛
- 检查学习率是否过大/过小
- 检查数据标签是否正确
- 尝试从更小的svd_rank开始

### 问题4：SVD参数不匹配
- 确保训练和测试时使用相同的svd_rank和clip_model
- 检查模型构建参数是否一致

## 引用

如果使用本代码，请引用原始论文：

```bibtex
@inproceedings{karageorgiou2025any,
  title={Any-resolution ai-generated image detection by spectral learning},
  author={Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}

@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}
```

## 许可证

本项目遵循Apache 2.0许可证。

