"""
基于Accelerate的深度学习引擎类
"""

import logging
import math
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import accelerate
import numpy as np
import torch
import torch.onnx as onnx
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .callback import Callback
from .model.base import Model
from .utils import collate_fn, get_newest_ckpt
from ..engine.base import Engine
from ..utils.helper import DummyClass
from ..utils.io import get_logger, reset_logger


class LossAggregator:
    """
    以均值的方式聚合loss字典

    loss字典中的每一个键值对应一个不同的loss
    """

    def __init__(self, ) -> None:
        self.num: int = 0
        self.loss: Dict[str, float] = defaultdict(float)

    def reset(self, ) -> None:
        """
        重置
        """
        self.num = 0
        self.loss = defaultdict(float)

    def update(self, loss: Dict[str, torch.Tensor]) -> None:
        """
        更新
        """
        self.num += 1
        for k, v in loss.items():
            self.loss[k] += (v.item() - self.loss[k]) / self.num


class AccelerateEngine(Engine):
    """
    基于Accelerate的深度学习引擎类
    """

    params: Dict[str, Any] = {
        "seed": 42,

        "lr": 0.001,
        "weight_decay": 0.0,

        "batch_size": 128,
        "n_worker": 4,
        "pin_memory": False,
        "prefetch_factor": 1,
        "persistent_workers": True,

        "epoch": 100,
        "n_grad_acc_step": 1,
        "clip_norm": 5.0,
    
        "n_val_per_epoch": 1,
    }

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        callbacks: List[Callback] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> None:
        _params: Dict[str, Any] = self.params.copy()
        if params is not None:
            _params.update(params)
        self.params = _params

        # 初始化accelerator
        accelerate.utils.set_seed(self.params["seed"])
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.params["n_grad_acc_step"],
            dataloader_config=accelerate.utils.DataLoaderConfiguration(
                non_blocking=True
            )
        )

        # 初始化文件夹, 初始化logger和tensorboard
        self.folder: str = folder
        if self.accelerator.is_main_process:
            if create_folder:
                # 移除旧文件夹并创建
                if os.path.exists(folder):
                    shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder)

            self.logger: logging.Logger = get_logger(folder)
            self.logger.info(f"params are {self.params}")
            self.writer = SummaryWriter(folder)
        else:
            self.logger = DummyClass()
            self.writer = DummyClass()

        # 注册回调
        self.callbacks: List[Callback] = []
        if callbacks is not None:
            self.callbacks.extend(callbacks)

        # 数据
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None

        # 模型
        self.unprepared_model: Model = None
        self.model = None
        self.optimizer: Optimizer = None

        # 运行变量
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.stop_train: bool = False

        # 当前训练步或验证步的输入输出
        self.data: Dict[str, Any] = {}
        self.output: Dict[str, torch.Tensor] = {}

        # 验证或测试结束后聚合的output和metric
        self.whole_output: Dict[str, torch.Tensor] = {}
        self.metric: Dict[str, float] = {}

    def get_model(self, ) -> Model:
        """
        从多卡封装的模型中取出原始module
        """
        return self.model.module if hasattr(
            self.model, "module"
        ) else self.model

    def set_model(self, model: Model) -> None:
        """
        设置模型
        """
        self.unprepared_model = model
        self.callbacks.append(model)
        self.optimizer = model.get_optimizer(self)

    def load_model(self, ckpt: str | None = None) -> None:
        """
        加载模型
        """
        self.accelerator.wait_for_everyone()

        if ckpt is None:
            ckpt = get_newest_ckpt(self.folder)

        state = torch.load(os.path.join(
            self.folder, ckpt
        ), map_location="cpu")
        self.unprepared_model.load_state_dict(state, strict=True)

    def set_dataset(
        self,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        """
        设置数据集

        数据集中的batch数据以字典的方式存放, 必须包含idx键值, 代表样本顺序
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train(self, ) -> None:
        """
        训练
        """
        torch.cuda.empty_cache()

        # 设置dataloader
        dataloader_config: Dict[str, Any] = {
            "batch_size": self.params["batch_size"],
            "collate_fn": collate_fn,
            "num_workers": self.params["n_worker"],
            "pin_memory": self.params["pin_memory"],
        }
        if dataloader_config["num_workers"] > 0:
            dataloader_config["prefetch_factor"] = self.params[
                "prefetch_factor"
            ]
            dataloader_config["persistent_workers"] = self.params[
                "persistent_workers"
            ]

        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            **dataloader_config,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            **dataloader_config,
        )

        # accelerator处理变量
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader
        ) = (
            self.accelerator.prepare(
                self.unprepared_model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
            )
        )

        # 计算评估步数, 这里的步数指模型同步梯度的步数
        # 向上取整是因为dataloader末尾一定会同步一次梯度
        opt_step: int = max(1, math.ceil(
            len(self.train_dataloader) / self.params["n_grad_acc_step"]
        ))
        val_step = max(1, opt_step // self.params["n_val_per_epoch"])

        # 训练循环
        train_loss_agg = LossAggregator()
        self._callback("on_train_start")

        for e in tqdm(
            range(self.params["epoch"]),
            disable=not self.accelerator.is_main_process,
        ):
            # 终止训练
            if self.stop_train:
                break

            self.current_epoch = e
            self._callback("on_epoch_start")

            for data in tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_main_process,
            ):
                # 终止训练
                if self.stop_train:
                    break
                
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    self._callback("on_train_step_start")

                    # 正向
                    self.data = data
                    self.output = self.model(data)
                    loss: Dict[str, torch.Tensor] = self.model.loss_func(self)
                    train_loss_agg.update(loss)

                    self.accelerator.backward(loss["loss"])
                    if (
                        self.accelerator.sync_gradients
                        and self.params["clip_norm"] != float("inf")
                    ):
                        # 只在同步梯度时做梯度裁剪
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.params["clip_norm"],
                        )

                    # 反向
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        # 只有在同步梯度时才触发验证
                        self.current_step += 1
                        self._callback("on_train_step_end")

                        # 验证
                        if self.current_step % val_step == 0:
                            self._callback("on_val_start")

                            # 正向
                            self.whole_output, val_loss_agg = self.predict(
                                self.val_dataloader, compute_loss=True
                            )
                            self.metric = self.model.metric_func(
                                self, self.val_dataset
                            )

                            # 记录
                            self._record(train_loss_agg.loss, prefix="train")
                            train_loss_agg.reset()
                            self._record(val_loss_agg.loss, prefix="val")
                            self._record(self.metric, prefix="val")

                            self._callback("on_val_end")

            self._callback("on_epoch_end")

        self._callback("on_train_end")

    def _callback(self, name: str) -> None:
        """
        按顺序执行回调
        """
        for callback in self.callbacks:
            getattr(callback, name)(self)

    def _record(self, value: Dict[str, Any], prefix: str) -> None:
        """
        将一个字典的值写入到log和tensorboard中
        """
        self.logger.info(
            f"epoch {self.current_epoch} step {self.current_step} "
            f"{prefix} {value}"
        )

        for k, v in value.items():
            self.writer.add_scalar(f"{prefix}_{k}", v, self.current_step)

    def test(self, ) -> None:
        """
        计算测试集指标
        """
        torch.cuda.empty_cache()

        # 加载数据集
        dataloader_config: Dict[str, Any] = {
            "batch_size": self.params["batch_size"],
            "collate_fn": collate_fn,
            "num_workers": self.params["n_worker"],
            "pin_memory": self.params["pin_memory"],
        }
        if dataloader_config["num_workers"] > 0:
            dataloader_config["prefetch_factor"] = self.params[
                "prefetch_factor"
            ]
            dataloader_config["persistent_workers"] = self.params[
                "persistent_workers"
            ]

        self.test_dataloader = DataLoader(
            self.test_dataset,
            **dataloader_config,
        )

        self.model, self.test_dataloader = self.accelerator.prepare(
            self.unprepared_model, self.test_dataloader,
        )

        # 计算指标
        self.whole_output, _ = self.predict(
            self.test_dataloader, compute_loss=False
        )
        self.metric: Dict[str, float] = self.model.metric_func(
            self, self.test_dataset
        )
        self.logger.info(f"test metric: {self.metric}")

    def predict(
        self,
        dataloader: DataLoader,
        compute_loss: bool,
    ) -> Tuple[Dict[str, torch.Tensor], LossAggregator]:
        """
        对指定dataloader推理得到输出

        通过gather_for_metrics聚合多机输出, 通过idx去重和排序

        注意这里有个假设是聚合后的output占用显存较小, 默认它们都会在gpu中聚合
        """
        self.model.eval()

        loss_agg = LossAggregator()
        idx_list: List[torch.Tensor] = []
        output_list: List[Dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for data in tqdm(
                dataloader,
                disable=not self.accelerator.is_main_process,
            ):
                self.data = data
                self.output = self.model(data)

                if compute_loss:
                    loss_agg.update(self.model.loss_func(self))

                idx_list.append(data["idx"].reshape(-1))
                output_list.append(self.output)

            idx: torch.Tensor = torch.cat(idx_list, axis=0)
            idx = self.accelerator.gather_for_metrics(idx)

            whole_output: Dict[str, torch.Tensor] = {}
            for k in output_list[0]:
                whole_output[k] = torch.cat(
                    [output_list[i][k] for i in range(len(output_list))
                ], axis=0)
                whole_output[k] = self.accelerator.gather_for_metrics(
                    whole_output[k]
                )

            # 根据idx对output调整顺序和去重并放置到cpu中
            perm: torch.Tensor = torch.argsort(idx)
            idx_sorted: torch.Tensor = idx.index_select(0, perm)
            mask = torch.ones_like(idx_sorted, dtype=torch.bool)
            mask[1: ] = (idx_sorted[1: ] != idx_sorted[: -1])

            for k in whole_output:
                whole_output[k] = whole_output[k].index_select(
                    0, perm[mask]
                ).cpu()
            return whole_output, loss_agg

    def run(self, task: str) -> None:
        """
        执行模型自定义任务的接口
        """
        getattr(self.model, task)(self, )

    def close(self, ) -> None:
        """
        主动调用以释放资源
        """
        if self.accelerator.is_main_process:
            reset_logger(self.logger)
            self.writer.close()
