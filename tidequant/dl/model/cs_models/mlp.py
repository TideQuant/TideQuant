"""
简单MLP模型
"""

from typing import Any, Dict, List

import torch
import numpy as np
import pandas as pd
from torch import nn

from ...ops import rank, nanmedian
from ..base import CSModel



class Preprocessor(nn.Module):
    """
    因子输入预处理类
    """

    def __init__(self, stats_csv_file: str, return_list: List[str] = ["x"]) -> None:
        self.stats: pd.DataFrame = pd.read_csv(stats_csv_file, index_col=0)

        

    def forward(self, x: torch.Tensor) -> None:


    def min_max(self, ) -> None:
    
    def rank(self, ) -> None:





class MLP(CSModel):

    def __init__(
        self,
        dim: int,
        output_dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
        use_mix_loss: bool = False,
        with_rank: bool = False,
    ) -> None:
        super().__init__()

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.dim: int = dim
        self.raw_dim: int = dim

        # rank预处理
        self.with_rank: bool = with_rank
        if self.with_rank:
            self.rank_x_slice: slice | list = slice(None)
            self.dim += self.raw_dim

        self.linear = nn.Sequential(
            nn.Linear(self.dim, 2048),
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

            nn.Linear(128, output_dim),
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

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 预处理x: (b, 1, n, d)
        x: torch.Tensor = data["x"].squeeze(1)
        x = torch.where(
            torch.isnan(x),
            # self.x_median.reshape(1, 1, -1),
            nanmedian(x, dim=1, keepdim=True)[0],
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0

        if self.with_rank:
            x = torch.cat([
                x, rank(data["x"][:, :, self.rank_x_slice], dim=1)
            ], dim=-1)

        # 推理得到输出
        b, n, d = x.shape
        y_pred = self.linear(x.reshape(-1, d)).reshape(b, n, -1)
        
        # 对输出标准化
        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std: np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}
