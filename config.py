# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Any

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.VAL_BATCH_SIZE = None
_C.DATA.TEST_BATCH_SIZE = None
_C.DATA.DATA_PATH = ''
_C.DATA.CSV_ROOT = ''
_C.DATA.TEST_DATA_PATH = []
_C.DATA.TEST_DATA_CSV_ROOT = []
_C.DATA.LMDB_PATH = None
_C.DATA.DATASET = 'imagenet'
_C.DATA.IMG_SIZE = 224
_C.DATA.MIN_CROP_SCALE = 0.2
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 24
_C.DATA.PREFETCH_FACTOR = 2
_C.DATA.VAL_PREFETCH_FACTOR = None
_C.DATA.TEST_PREFETCH_FACTOR = None

_C.DATA.FILTER_TYPE = 'mfm'
_C.DATA.SAMPLE_RATIO = 0.5
_C.DATA.MASK_RADIUS1 = 16
_C.DATA.MASK_RADIUS2 = 999
_C.DATA.SR_FACTOR = 8
_C.DATA.BLUR = CN()
_C.DATA.BLUR.KERNEL_SIZE = [7, 9, 11, 13, 15, 17, 19, 21]
_C.DATA.BLUR.KERNEL_LIST = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'sinc']
_C.DATA.BLUR.KERNEL_PROB = [0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1]
_C.DATA.BLUR.SIGMA_X = [0.2, 3]
_C.DATA.BLUR.SIGMA_Y = [0.2, 3]
_C.DATA.BLUR.ROTATE_ANGLE = [-3.1416, 3.1416]
_C.DATA.BLUR.BETA_GAUSSIAN = [0.5, 4]
_C.DATA.BLUR.BETA_PLATEAU = [1, 2]
_C.DATA.NOISE = CN()
_C.DATA.NOISE.TYPE = ['gaussian', 'poisson']
_C.DATA.NOISE.PROB = [0.5, 0.5]
_C.DATA.NOISE.GAUSSIAN_SIGMA = [1, 30]
_C.DATA.NOISE.GAUSSIAN_GRAY_NOISE_PROB = 0.4
_C.DATA.NOISE.POISSON_SCALE = [0.05, 3]
_C.DATA.NOISE.POISSON_GRAY_NOISE_PROB = 0.4
_C.DATA.AUGMENTED_VIEWS = 1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'vit'
_C.MODEL_WEIGHTS = "mfm"
_C.MODEL.NAME = 'pretrain'
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.SID_DROPOUT = 0.5
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.REQUIRED_NORMALIZATION = "imagenet"
_C.MODEL.SID_APPROACH = "single_extraction"
_C.MODEL.RESOLUTION_MODE = "fixed"
_C.MODEL.FEATURE_EXTRACTION_BATCH = None

_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.INIT_VALUES = 0.1
_C.MODEL.VIT.USE_APE = True
_C.MODEL.VIT.USE_FPE = False
_C.MODEL.VIT.USE_RPB = False
_C.MODEL.VIT.USE_SHARED_RPB = False
_C.MODEL.VIT.USE_MEAN_POOLING = False
_C.MODEL.VIT.DECODER = CN()
_C.MODEL.VIT.DECODER.EMBED_DIM = 512
_C.MODEL.VIT.DECODER.DEPTH = 0
_C.MODEL.VIT.DECODER.NUM_HEADS = 16

_C.MODEL.VIT.FEATURES_PROCESSOR = "rine"
_C.MODEL.VIT.USE_INTERMEDIATE_LAYERS = False
_C.MODEL.VIT.INTERMEDIATE_LAYERS = [2, 5, 8, 11]
_C.MODEL.VIT.PROJECTION_DIM = 1024
_C.MODEL.VIT.PROJECTION_LAYERS = 2
_C.MODEL.VIT.PATCH_PROJECTION = False
_C.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE = False
_C.MODEL.VIT.PATCH_POOLING = "mean"

_C.MODEL.FRE = CN()
_C.MODEL.FRE.MASKING_RADIUS = 16
_C.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE = "gelu"
_C.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH = False
_C.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY = False

_C.MODEL.PATCH_VIT = CN()
_C.MODEL.PATCH_VIT.PATCH_STRIDE = 224
_C.MODEL.PATCH_VIT.NUM_HEADS = 12
_C.MODEL.PATCH_VIT.ATTN_EMBED_DIM = 1536
_C.MODEL.PATCH_VIT.MINIMUM_PATCHES = 1

_C.MODEL.CLS_HEAD = CN()
_C.MODEL.CLS_HEAD.MLP_RATIO = 4

_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.LAYERS = [3, 4, 6, 3]
_C.MODEL.RESNET.IN_CHANS = 3

_C.MODEL.RECOVER_TARGET_TYPE = 'normal'
_C.MODEL.FREQ_LOSS = CN()
_C.MODEL.FREQ_LOSS.LOSS_GAMMA = 1.
_C.MODEL.FREQ_LOSS.MATRIX_GAMMA = 1.
_C.MODEL.FREQ_LOSS.PATCH_FACTOR = 1
_C.MODEL.FREQ_LOSS.AVE_SPECTRUM = False
_C.MODEL.FREQ_LOSS.WITH_MATRIX = False
_C.MODEL.FREQ_LOSS.LOG_MATRIX = False
_C.MODEL.FREQ_LOSS.BATCH_MATRIX = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3e-4
_C.TRAIN.WARMUP_LR = 2.5e-7
_C.TRAIN.MIN_LR = 2.5e-6
_C.TRAIN.CLIP_GRAD = 3.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.MODE = "supervised"
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
_C.TRAIN.SCALE_LR = False
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.LOSS = "bce_supcont"
_C.TRAIN.TRIPLET_LOSS_MARGIN = 0.5
_C.TRAIN.LAYER_DECAY = 1.0

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.MIN_CROP_AREA = 0.2
_C.AUG.MAX_CROP_AREA = 1.0
_C.AUG.HORIZONTAL_FLIP_PROB = 0.5
_C.AUG.VERTICAL_FLIP_PROB = 0.5
_C.AUG.ROTATION_PROB = 0.5
_C.AUG.ROTATION_DEGREES = 90
_C.AUG.GAUSSIAN_BLUR_PROB = 0.5
_C.AUG.GAUSSIAN_BLUR_LIMIT = (3, 9)
_C.AUG.GAUSSIAN_BLUR_SIGMA = (0.01, 0.5)
_C.AUG.GAUSSIAN_NOISE_PROB = 0.5
_C.AUG.JPEG_COMPRESSION_PROB = 0.5
_C.AUG.JPEG_MIN_QUALITY = 50
_C.AUG.JPEG_MAX_QUALITY = 100
_C.AUG.WEBP_COMPRESSION_PROB = .0
_C.AUG.WEBP_MIN_QUALITY = 50
_C.AUG.WEBP_MAX_QUALITY = 100
_C.AUG.COLOR_JITTER = .0
_C.AUG.COLOR_JITTER_BRIGHTNESS_RANGE = (0.8, 1.2)
_C.AUG.COLOR_JITTER_CONTRAST_RANGE = (0.8, 1.2)
_C.AUG.COLOR_JITTER_SATURATION_RANGE = (0.8, 1.2)
_C.AUG.COLOR_JITTER_HUE_RANGE = (-0.1, 0.1)
_C.AUG.SHARPEN_PROB = .0
_C.AUG.SHARPEN_ALPHA_RANGE = (0.01, 0.4)
_C.AUG.SHARPEN_LIGHTNESS_RANGE = (0.95, 1)
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.BLUR_PROB = 0.25
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.MAX_SIZE: Optional[int] = None
_C.TEST.ORIGINAL_RESOLUTION = False
_C.TEST.VIEWS_GENERATION_APPROACH = None
_C.TEST.VIEWS_REDUCTION_APPROACH = "mean"
_C.TEST.EXPORT_IMAGE_PATCHES = True
_C.TEST.GAUSSIAN_BLUR = False
_C.TEST.GAUSSIAN_BLUR_KERNEL_SIZE = 3
_C.TEST.GAUSSIAN_NOISE = False
_C.TEST.GAUSSIAN_NOISE_SIGMA = 1.0
_C.TEST.JPEG_COMPRESSION = False
_C.TEST.JPEG_QUALITY = 100
_C.TEST.WEBP_COMPRESSION = False
_C.TEST.WEBP_QUALITY = 100
_C.TEST.SCALE = False
_C.TEST.SCALE_FACTOR = 1.0

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 10
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0
_C.PRETRAINED = ''


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    # 支持"cfg"或"config_path"作为键名
    cfg_file = args.get("cfg") or args.get("config_path")
    if cfg_file:
        _update_config_from_file(config, cfg_file)

    config.defrost()
    if "opts" in args:
        options: list[Any] = []
        for (k, v) in args["opts"]:
            options.append(k)
            options.append(eval(v))
        config.merge_from_list(options)

    def _check_args(name):
        if name in args and args[name]:
            return True
        return False

    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args["batch_size"]
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args["data_path"]
    if _check_args('csv_root_dir'):
        config.DATA.CSV_ROOT = args["csv_root_dir"]
    if _check_args("lmdb_path"):
        config.DATA.LMDB_PATH = args["lmdb_path"]
    if _check_args('resume'):
        config.MODEL.RESUME = args["resume"]
    if _check_args('pretrained'):
        config.PRETRAINED = args["pretrained"]
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args["accumulation_steps"]
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args["amp_opt_level"]
    if _check_args('output'):
        config.OUTPUT = args["output"]
    if _check_args('tag'):
        config.TAG = args["tag"]
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('test_csv'):
        config.DATA.TEST_DATA_PATH = args["test_csv"]
    if _check_args('test_csv_root'):
        config.DATA.TEST_DATA_CSV_ROOT = args["test_csv_root"]
    if _check_args('learning_rate'):
        config.TRAIN.BASE_LR = args["learning_rate"]
    if _check_args('resize_to'):
        config.TEST.MAX_SIZE = args["resize_to"]
    if _check_args("local_rank"):
        config.LOCAL_RANK = args["local_rank"]
    if _check_args("data_workers"):
        config.DATA.NUM_WORKERS = args["data_workers"]
    if _check_args("disable_pin_memory"):
        config.PIN_MEMORY = False
    if _check_args("data_prefetch_factor"):
        config.DATA.PREFETCH_FACTOR = args["data_prefetch_factor"]
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)
    return config


def get_custom_config(cfg):
    config = _C.clone()
    _update_config_from_file(config, cfg)
    return config
