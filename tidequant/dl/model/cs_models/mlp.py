"""
多层感知机模型
"""

from typing import Dict, List

import torch
import pandas as pd
from torch import nn

from ...ops import nanmedian, rank
from ..base import CSModel


class Preprocessor(nn.Module):
    """
    因子输入预处理类

    根据return list返回预处理后的因子
    """

    def __init__(
        self,
        x_fields: List[str],
        return_list: List[str],
        stats_csv_file: str = "",
    ) -> None:
        super().__init__()

        stats: pd.DataFrame | None = pd.read_csv(
            stats_csv_file, index_col=0
        ) if stats_csv_file != "" else None
        self.return_list: List[str] = return_list
        assert len(self.return_list) >= 1
    
        if "winsor_min_max" in return_list:
            self.register_buffer(
                "x_min", torch.tensor(stats.loc[x_fields]["x_1"].values)
            )
            self.register_buffer(
                "x_median", torch.tensor(stats.loc[x_fields]["x_50"].values)
            )
            self.register_buffer(
                "x_max", torch.tensor(stats.loc[x_fields]["x_99"].values)
            )

        # TODO: 之后的预处理方法增加的维数可能不同
        self.output_dim: int = len(x_fields) * len(self.return_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入为(batch, seq_len, n_ticker, n_field)

        输出为(batch, seq_len, n_ticker, -1)
        """
        x_list: List[torch.Tensor] = []
        for name in self.return_list:
            if name == "winsor_min_max":
                x_list.append(self.winsor_min_max(x))
            elif name == "rank":
                x_list.append(self.rank(x))
            else:
                raise RuntimeError(f"Unknown preprocessor: {name}")
        return torch.cat(x_list, dim=-1)

    def winsor_min_max(self, x: torch.Tensor) -> torch.Tensor:
        """
        winsorized min-max归一化
        """
        assert x.ndim == 4
        x = torch.where(
            torch.isnan(x),
            # self.x_median.reshape(1, 1, 1, -1),
            nanmedian(x, dim=-2, keepdim=True)[0],
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0
        return x

    def rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        排名归一化
        """
        assert x.ndim == 4
        return rank(x, dim=-2)


class MLP(CSModel):

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        return_list: List[str],
        use_mix_loss: bool = False,
        stats_csv_file: str = "",
    ) -> None:
        super().__init__(
            x_fields=x_fields,
            y_fields=y_fields,
            use_mix_loss=use_mix_loss,
        )

        self.preprocessor = Preprocessor(
            x_fields=x_fields,
            return_list=return_list,
            stats_csv_file=stats_csv_file,
        )
        self.linear = nn.Sequential(
            nn.Linear(self.preprocessor.output_dim, 2048),
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
        x: torch.Tensor = self.preprocessor(data["x"])

        # 推理得到输出
        b, t, n, d = x.shape
        assert t == 1
        y_pred: torch.Tensor = self.linear(x.reshape(-1, d)).reshape(b, n, -1)

        # 对输出标准化
        mean: torch.Tensor = y_pred.mean(dim=-2, keepdim=True)
        std: torch.Tensor = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}
