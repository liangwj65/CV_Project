# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
数据集模块 - 复用SPAI的数据处理
"""

import sys
from pathlib import Path

from data.data_finetune import CSVDataset, build_loader_test as spai_build_loader_test
from data import build_loader as spai_build_loader


def build_loader(config, logger, is_train=True):
    """构建数据加载器"""
    dataset_train, dataset_val, dataloader_train, dataloader_val, mixup_fn = spai_build_loader(
        config, logger, is_pretrain=False
    )

    if mixup_fn is not None:
        logger.warning("当前训练流程未使用mixup_fn，如需使用请在train.py中显式处理。")

    return dataloader_train if is_train else dataloader_val


def build_loader_test(config, logger, split="test"):
    """构建测试数据加载器"""
    return spai_build_loader_test(config, logger, split=split)

