import datetime
from typing import Dict, Union

import easytorch

def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0, inference: bool = False, date_inference: datetime = None):
    """Extended easytorch launch_training.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    # pre-processing of some possible future features, such as:
    # registering model, runners.
    # config checking
    # launch training based on easytorch
    try:
        return easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank, inference=inference, date_inference=date_inference or datetime.datetime(2024, 6, 11, 0, 35))
    except TypeError as e:
        if "launch_training() got an unexpected keyword argument" in repr(e):
            # NOTE: for earlier easytorch version
            easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
        else:
            raise e
