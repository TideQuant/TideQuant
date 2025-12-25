"""
多层感知机模型
"""

from typing import Dict, List

import torch
import pandas as pd
from torch import nn

from .preprocessor import Preprocessor
from ..base import CSModel
from ...ops import nanmedian, rank



class ConcatMLP(CSModel):
    """
    将多个预处理concat起来输入的MLP
    """

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        prep_names: List[str],
        stats_csv_file: str = "",
    ) -> None:
        super().__init__(
            x_fields=x_fields,
            y_fields=y_fields,
        )

        self.prep_names: List[str] = prep_names
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


class WeightSumMLP(CSModel):
    """
    将多个预处理独立过MLP, 通过可学习的权重加权融合
    """

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        prep_names: List[str],
        stats_csv_file: str = "",
    ) -> None:
        super().__init__(
            x_fields=x_fields,
            y_fields=y_fields,
        )

        self.prep_names: List[str] = prep_names
        self.weights: torch.Tensor = nn.Parameter(torch.ones(len(prep_names)))
        self.preprocessor = Preprocessor(
            x_fields=x_fields,
            stats_csv_file=stats_csv_file,
        )
        self.linears = nn.ModuleList(nn.Sequential(
            # TODO: 不同预处理增加的维度可能不同
            nn.Linear(len(x_fields), 2048),
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
        ) for _ in self.prep_names)

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
        x: torch.Tensor = data["x"]
        b, t, n, d = x.shape
        assert t == 1

        y_pred_list: List[torch.Tensor] = []
        for i, linear in enumerate(self.linears):
            x_prep: torch.Tensor = self.preprocessor(x, self.prep_names[i])
            y_pred_list.append(
                linear(x_prep.reshape(-1, d)).reshape(b, n, -1)
            )

        y_pred: torch.Tensor = torch.stack(y_pred_list, dim=-1)
        y_pred = (y_pred * self.weights).sum(dim=-1)

        # 对输出标准化
        mean: torch.Tensor = y_pred.mean(dim=-2, keepdim=True)
        std: torch.Tensor = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}
