"""
多层感知机模型
"""

from typing import Dict, List
from functools import partial

import torch
from torch import nn

from .preprocessor import Preprocessor
from ..base import CSModel
from ...engine import AccelerateEngine
from ...ops import abs_weight_ic_loss, tail_weight_ic_loss


class WICConcatMLP(CSModel):
    """
    将多个预处理concat起来输入的MLP
    """

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        prep_names: List[str],
        stats_csv_file: str,
        quantile: float = 0.0,
        weight: float = 3.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(
            x_fields=x_fields,
            y_fields=y_fields,
        )

        self.prep_names: List[str] = prep_names
        self.quantile: float = quantile
        self.weight: float = weight
        self.scale: float = scale

        self.preprocessor = Preprocessor(
            x_fields=x_fields,
            stats_csv_file=stats_csv_file,
        )
        self.linear = nn.Sequential(
            # TODO: 不同预处理增加的维度可能不同
            nn.Linear(len(x_fields) * len(prep_names), 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.8),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, len(self.y_fields)),
        )

        self._init_weight()

    def _init_weight(self, ) -> None:
        """
        自定义初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=0.01,
                        mode="fan_in",
                        nonlinearity="leaky_relu",
                    )
                    nn.init.constant_(m.bias, 0)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_list: List[torch.Tensor] = []
        for name in self.prep_names:
            x_list.append(self.preprocessor(data["x"], name))

        x: torch.Tensor = torch.concat(x_list, dim=-1)

        # 推理得到输出
        b, t, n, d = x.shape
        assert t == 1
        y_pred: torch.Tensor = self.linear(x.reshape(-1, d)).reshape(b, n, -1)

        # 对输出标准化
        mean: torch.Tensor = y_pred.mean(dim=-2, keepdim=True)
        std: torch.Tensor = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}

    def loss_func(self, engine: AccelerateEngine) -> torch.Tensor:
        """
        计算损失函数
        """
        if self.quantile > 0:
            loss_func = partial(
                tail_weight_ic_loss,
                quantile=self.quantile,
                tail_weight=self.weight,
            )
        else:
            loss_func = partial(
                abs_weight_ic_loss,
                scale=self.scale,
                max_weight=self.weight,
            )

        return {"loss": loss_func(
            engine.output["y_pred"], engine.data["y"], dim=-2,
        )}
