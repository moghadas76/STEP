import os
import sys


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj, load_pkl

from .step_arch import STDMAE
from .step_runner import STEPRunner
from .step_loss import step_loss_tiny
from .step_data import ForecastingDataset



CFG = EasyDict()

CFG.MD5 = "f8cbc6b5dc005fca0774a226a394b6d2_std"
# Resume

# ================= general ================= #
CFG.DESCRIPTION = "STEP(METR-LA) configuration"
CFG.RUNNER = STEPRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.DATASET_ARGS = {"seq_len": 288 * 7}
CFG.GPU_NUM = 2

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STDMAE"
CFG.MODEL.ARCH = STDMAE
adj_mx, _ = load_adj("datasets/METR-LA/adj_mx.pkl", "doubletransition")
CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_tmae_path": "checkpoints/Mask_200/439add74a377d6a14f373b75169eaba6/Mask_best_val_MAE.pt",
    "pre_trained_smae_path": "checkpoints/Mask_200/439add74a377d6a14f373b75169eaba6/Mask_best_val_MAE.pt",
    "mask_args": {
        "patch_size": 12,
        "in_channel": 1,
        "embed_dim": 96,
        "num_heads": 4,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "mask_ratio": 0.25,
        "encoder_depth": 4,
        "decoder_depth": 1,
        "mode": "forecasting",
    },
    "backend_args": {
        "num_nodes": 207,
        "supports": [torch.tensor(adj) for adj in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": 12,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    },
}
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = step_loss_tiny
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 18, 36, 54, 72], "gamma": 0.5}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 3.0}
CFG.TRAIN.NUM_EPOCHS = 300
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints", "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# CFG.TRAIN.NUM_EPOCHS = 150
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True
# curriculum learning
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
