"""
因子预处理类
"""

from typing import List

import torch
import pandas as pd
from torch import nn

from ...ops import nanmedian, rank


class Preprocessor(nn.Module):
    """
    因子输入预处理类

    根据传入的name返回不同的预处理结果
    """

    def __init__(
        self,
        x_fields: List[str],
        stats_csv_file: str,
    ) -> None:
        super().__init__()

        stats: pd.DataFrame = pd.read_csv(
            stats_csv_file, index_col=0
        )
        self.register_buffer(
            "x_1",
            torch.tensor(
                stats.loc[x_fields]["x_1"].values, dtype=torch.float32
            )
        )
        self.register_buffer(
            "x_25",
            torch.tensor(
                stats.loc[x_fields]["x_25"].values, dtype=torch.float32
            )
        )
        self.register_buffer(
            "x_50",
            torch.tensor(
                stats.loc[x_fields]["x_50"].values, dtype=torch.float32
            )
        )
        self.register_buffer(
            "x_75",
            torch.tensor(
                stats.loc[x_fields]["x_75"].values, dtype=torch.float32
            )
        )
        self.register_buffer(
            "x_99",
            torch.tensor(
                stats.loc[x_fields]["x_99"].values, dtype=torch.float32
            )
        )

    def forward(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "winsor_min_max":
            return self.winsor_min_max(x)

        if name == "rank":
            return self.rank(x)

        if name == "cs_robust_zscore":
            return self.cs_robust_zscore(x)
        
        if name == "robust_zscore":
            return self.robust_zscore(x)

        if name == "log1p":
            return self.log1p(x)

        raise ValueError(f"{name} is not supported")

    def winsor_min_max(self, x: torch.Tensor) -> torch.Tensor:
        """
        winsorized min-max归一化
        """
        assert x.ndim == 4

        x = torch.where(
            torch.isnan(x),
            self.x_50.reshape(1, 1, 1, -1),
            # nanmedian(x, dim=-2, keepdim=True)[0],
            x,
        )
        x = (x - self.x_1) / (self.x_99 - self.x_1)
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

    def cs_robust_zscore(self, x: torch.Tensor) -> torch.Tensor:
        """
        鲁棒的截面zscore
        """
        assert x.ndim == 4

        q25: torch.Tensor = torch.nanquantile(x, 0.25, dim=-2, keepdim=True)
        q50: torch.Tensor = torch.nanquantile(x, 0.50, dim=-2, keepdim=True)
        q75: torch.Tensor = torch.nanquantile(x, 0.75, dim=-2, keepdim=True)
        iqr: torch.Tensor = q75 - q25
        x = (x - q50) / (1.35 * iqr)

        x[torch.isnan(x)] = 0.0
        return torch.clamp(x, min=-3, max=3)

    def robust_zscore(self, x: torch.Tensor) -> torch.Tensor:
        """
        鲁棒的整体zscore
        """
        assert x.ndim == 4

        iqr: torch.Tensor = self.x_75 - self.x_25
        x = (x - self.x_50) / (1.35 * iqr)
        
        x[torch.isnan(x)] = 0.0
        return torch.clamp(x, min=-3, max=3)

    def log1p(self, x: torch.Tensor) -> torch.Tensor:
        """
        log1p预处理
        """
        assert x.ndim == 4

        x = torch.where(
            torch.isnan(x),
            self.x_50.reshape(1, 1, 1, -1),
            # nanmedian(x, dim=-2, keepdim=True)[0],
            x,
        )
        x = torch.where(x >= 0, torch.log(x + 1), -torch.log(-x + 1))
        x_1: torch.Tensor = torch.where(
            self.x_1 >= 0, torch.log(self.x_1 + 1), -torch.log(-self.x_1 + 1)
        )
        x_99: torch.Tensor = torch.where(
            self.x_99 >= 0, torch.log(self.x_99 + 1), -torch.log(-self.x_99 + 1)
        )
        x = (x - x_1) / (x_99 - x_1)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0
        return x
