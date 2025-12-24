"""
多层感知机模型
"""

from typing import Dict, List, Tuple

import torch
import pandas as pd
from torch import nn

from ..base import CSModel
from ...engine import AccelerateEngine
from ...ops import nanmedian, rank


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
                "x_min",
                torch.tensor(
                    stats.loc[x_fields]["x_1"].values, dtype=torch.float32
                )
            )
            self.register_buffer(
                "x_median",
                torch.tensor(
                    stats.loc[x_fields]["x_50"].values, dtype=torch.float32
                )
            )
            self.register_buffer(
                "x_max",
                torch.tensor(
                    stats.loc[x_fields]["x_99"].values, dtype=torch.float32
                )
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


class MLPWeightIC(CSModel):

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

    def loss_func(self, engine: AccelerateEngine) -> Dict[str, torch.Tensor]:
        _ic_loss, _ = weighted_ic_loss(
            engine.output["y_pred"], engine.data["y"]
        )
        return {"loss": _ic_loss}


def weighted_ic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_q: float = 0.9,
    tail_q: float = 0.1,
    head_weight: float = 2.0,
    tail_weight: float = 2.0,
    mid_weight: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    pred/target: (seq, N, d)
    在 N 维算加权IC, 对 seq 和 d 取平均, target 中无效位置已为 NaN
    返回: (loss, valid_slices)
    """
    assert pred.shape == target.shape, "pred/target 形状必须一致"
    assert pred.ndim == 3, "pred/target 必须是 (seq, N, d)"

    seq, _, d = pred.shape
    losses = []
    valid_slices = 0.0

    for t in range(seq):
        for k in range(d):
            p = pred[t, :, k]
            y = target[t, :, k]

            m = torch.isfinite(y) & torch.isfinite(p)
            if m.sum() < 2:
                continue

            p_m = p[m]
            y_m = y[m]
            n_valid = int(p_m.numel())

            # 用 y 的排序做分位 q \in [0,1]
            _, rank_idx = torch.sort(y_m, descending=False)
            ranks = torch.empty_like(y_m)
            ranks[rank_idx] = torch.arange(n_valid, device=y_m.device, dtype=y_m.dtype)
            q = ranks / max(n_valid - 1, 1)

            # 权重：头尾加权
            w = torch.ones_like(y_m) * mid_weight
            w[q >= head_q] = head_weight
            w[q <= tail_q] = tail_weight

            wsum = w.sum()
            if wsum <= eps:
                continue

            # 加权去中心
            p_hat = p_m - (p_m * w).sum() / wsum
            y_hat = y_m - (y_m * w).sum() / wsum

            # 加权相关（IC）
            num = (w * p_hat * y_hat).sum()
            den_p = torch.sqrt((w * p_hat * p_hat).sum())
            den_y = torch.sqrt((w * y_hat * y_hat).sum())
            den = den_p * den_y + eps

            ic = num / den
            if torch.isfinite(ic):
                losses.append(-ic)
                valid_slices += 1.0

    if len(losses) == 0:
        return (pred * 0.0).sum(), 0.0

    return torch.stack(losses).mean(), valid_slices
