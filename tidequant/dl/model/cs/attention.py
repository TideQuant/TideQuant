import math
from typing import Dict, List

import torch
from torch import nn

from .mlp import Preprocessor
from ..base import CSModel


class DiagAttention(nn.Module):
    """
    线性层均为对角矩阵的Self-Attention
    
    聚焦股票间因子驱动的线性关联
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        
        self.dim = dim
        self.q_diag = nn.Parameter(torch.ones(dim))
        self.k_diag = nn.Parameter(torch.ones(dim))
        self.v_diag = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, n, d)
        b, n, d = x.shape

        Q: torch.Tensor = x * self.q_diag
        K: torch.Tensor = x * self.k_diag
        V: torch.Tensor = x * self.v_diag
        
        logits: torch.Tensor = ((
            Q @ K.transpose(-1, -2)
        ) / math.sqrt(self.dim)).softmax(dim=-1)
        return logits @ V


class Attention(CSModel):

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
        self.attention = DiagAttention(self.preprocessor.output_dim)
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
        x = x.reshape(b, n, d)

        # 应用diag attention
        x = self.attention(x)
        y_pred: torch.Tensor = self.linear(x.reshape(-1, d)).reshape(b, n, -1)

        # 对输出标准化
        mean: torch.Tensor = y_pred.mean(dim=-2, keepdim=True)
        std: torch.Tensor = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}
