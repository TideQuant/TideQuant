"""
基于Accelerate Engine的回调类
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC
from typing import List, Literal, TYPE_CHECKING

import torch
from ray.air import session
from transformers import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from .engine import AccelerateEngine


class Callback(ABC):
    """
    回调类

    在实验过程中按顺序自动调用执行
    """

    def on_epoch_start(self, engine: AccelerateEngine) -> None:
        pass

    def on_epoch_end(self, engine: AccelerateEngine) -> None:
        pass

    def on_train_start(self, engine: AccelerateEngine) -> None:
        pass

    def on_train_end(self, engine: AccelerateEngine) -> None:
        pass

    def on_train_step_start(self, engine: AccelerateEngine) -> None:
        pass

    def on_train_step_end(self, engine: AccelerateEngine) -> None:
        pass

    def on_val_start(self, engine: AccelerateEngine) -> None:
        pass
    
    def on_val_end(self, engine: AccelerateEngine) -> None:
        pass


class EarlyStopSaver(Callback):
    """
    每次验证结束后根据metric早停和保存模型

    metric为None时代表每次都是最优模型

    除了支持按照指标早停, 也支持按最大评估次数早停
    """

    def __init__(
        self,
        metric_name: str | None = None,
        patience: int = 10,
        mode: Literal["min", "max"] = "max",
        tol: float = 0.0,
        max_n_val: int = sys.maxsize,
    ) -> None:
        self.metric_name: str | None = metric_name
        self.patience: int = patience
        self.mode: Literal["min", "max"] = mode
        self.tol: float = tol
        self.max_n_val: int = max_n_val

        self.best_metric: float | None = None
        self.current_patience: int = self.patience
        self.current_n_val: int = 0

    def _is_better(self, metric: float) -> bool:
        """
        返回新指标是否更优
        """
        if self.best_metric is None:
            return True

        if self.mode == "min":
            return metric < self.best_metric - self.tol
        elif self.mode == "max":
            return metric > self.best_metric + self.tol
        else:
            raise ValueError(f"{self.mode} must be in [min, max]")

    def on_val_end(self, engine: AccelerateEngine) -> None:
        """
        根据指标早停和保存模型
        """
        if (
            (self.metric_name is None) or
            self._is_better(engine.metric[self.metric_name])
        ):
            self.current_patience = self.patience
            self.best_metric = engine.metric.get(self.metric_name, None)

            engine.logger.info(
                f"epoch {engine.current_epoch} "
                f"step {engine.current_step} new best model"
            )

            # 保存最优模型
            if engine.accelerator.is_main_process:
                torch.save(
                    engine.get_model().state_dict(),
                    os.path.join(
                        engine.folder, f"model_{engine.current_step}.pth"
                    ),
                )
        else:
            # 触发早停逻辑
            self.current_patience -= 1
            if self.current_patience == 0:
                engine.logger.info("early stop")
                engine.stop_train = True

        # 达到最大评估次数会停止训练
        self.current_n_val += 1
        if self.current_n_val >= self.max_n_val:
            engine.logger.info(
                f"reach max_n_val {self.max_n_val}, early return"
            )
            engine.stop_train = True


class RayTuneReport(Callback):
    """
    每次验证结束后通过ray上报指标

    上报的是到目前为止的最优指标
    """

    def __init__(
        self,
        metric_name: str,
        mode: Literal["min", "max"] = "max",
        tol: float = 0.0,
    ) -> None:
        self.metric_name: str = metric_name
        self.mode: Literal["min", "max"] = mode
        self.tol: float = tol

        self.best_metric: float | None = None

    def _is_better(self, metric: float) -> bool:
        """
        返回新指标是否更优
        """
        if self.best_metric is None:
            return True

        if self.mode == "min":
            return metric < self.best_metric - self.tol
        elif self.mode == "max":
            return metric > self.best_metric + self.tol
        else:
            raise ValueError(f"{self.mode} must be in [min, max]")

    def on_val_end(self, engine: AccelerateEngine) -> None:
        """
        根据指标上报指标
        """
        if self._is_better(engine.metric[self.metric_name]):
            self.best_metric = engine.metric[self.metric_name]

        if engine.accelerator.is_main_process:
            session.report({self.metric_name: self.best_metric})


class WarmUpSchedule(Callback):
    """
    预热学习率调度器
    """

    def __init__(self, warmup_ratio: float = 0.0) -> None:
        self.warmup_ratio: float = warmup_ratio

    def on_train_start(self, engine: AccelerateEngine) -> None:
        total_step: int = math.ceil(
            len(engine.train_dataloader) / engine.params["n_grad_acc_step"]
        ) * engine.params["epoch"]
        engine.scheduler = get_linear_schedule_with_warmup(
            engine.optimizer,
            num_warmup_steps=int(total_step * self.warmup_ratio),
            num_training_steps=total_step,
        )

    def on_train_step_end(self, engine: AccelerateEngine) -> None:
        engine.scheduler.step()
