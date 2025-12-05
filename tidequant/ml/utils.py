"""
机器学习相关的工具函数
"""


import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    设置深度学习实验涉及到的所有随机数种子
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
