# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .data_finetune import build_loader_finetune, build_loader_test

def build_loader(config, logger, is_pretrain=False, is_test=False):
    if is_pretrain:
        raise NotImplementedError("Pretrain loader not implemented in fusion")
    elif is_test:
        return build_loader_test(config, logger)
    else:
        return build_loader_finetune(config, logger)
