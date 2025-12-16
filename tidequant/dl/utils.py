"""
深度学习相关的工具函数
"""

import random
import os
from typing import Any, Dict, List

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


def collate_fn(
    data_list: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor | List[Any]]:
    """
    将列表中的字典的相同字段合并为一个字典:
    1. numpy矩阵会被合并成tensor
    2. 非numpy矩阵会被合并成列表
    """

    data: Dict[str, List[Any]] = {}
    for item in data_list:
        for k, v in item.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k in data:
        if not isinstance(data[k][0], np.ndarray):
            continue

        if data[k][0].dtype == np.bool_:
            dtype = torch.bool
        elif data[k][0].dtype == np.float32:
            dtype = torch.float32
        elif data[k][0].dtype == np.int64:
            dtype = torch.int64
        else:
            raise ValueError(f"{data[k][0].dtype} is not supported")

        data[k] = np.stack(data[k], axis=0)
        data[k] = torch.tensor(data[k], dtype=dtype) 
    return data


def load_to_device(
    data: Dict[str, torch.Tensor | List[Any]],
    device: str,
) -> None:
    """
    将tensor矩阵加载到指定设备中
    """

    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(device, non_blocking=True)


def get_oldest_ckpt(folder: str) -> str:
    """
    从一个文件夹下面获取最老的checkpoint路径

    文件夹下的checkpoint需按照f"model{step}.pth"格式

    常用于推断模型训练的最优epoch
    """

    ckpts: List[str] = os.listdir(folder)
    ckpts = [ckpt for ckpt in ckpts if ckpt[: 6] == "model_"]
    return min(ckpts, key=lambda s: int(s.split('_')[1].split('.')[0]))


def get_newest_ckpt(folder: str) -> str:
    """
    从一个文件夹下面获取最新的checkpoint路径

    文件夹下的checkpoint需按照f"model{step}.pth"格式
    """

    ckpts: List[str] = os.listdir(folder)
    ckpts = [ckpt for ckpt in ckpts if ckpt[: 6] == "model_"]
    return max(ckpts, key=lambda s: int(s.split('_')[1].split('.')[0]))
