"""
用于因子选择的GFSNetwork

https://arxiv.org/pdf/2503.13304
"""

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch import nn

from .mlp import Preprocessor
from ..base import CSModel
from ...engine import AccelerateEngine
from ...ops import ic_loss


class GumbelSigmoidGate(nn.Module):
    """
    基于Gumbel Sigmoid的特征门控
    """

    def __init__(
        self,
        dim: int,
        tau: float = 2.0,
        min_tau: float = 0.3,
        tau_decay: float = 0.83,
        keep: float = 1.0,
        min_keep: float = 0.05,
        keep_decay: float = 0.6,
        noise: float = 1.0,
        min_noise: float = 0.0,
        noise_decay: float = 0.4,
    ) -> None:
        super().__init__()

        self.tau: float = tau
        self.min_tau: float = min_tau
        self.tau_decay: float = tau_decay
        self.keep: float = keep
        self.min_keep: float = min_keep
        self.keep_decay: float = keep_decay
        self.noise: float = noise
        self.min_noise: float = min_noise
        self.noise_decay: float = noise_decay

        self.logits = nn.Parameter(torch.zeros(dim))

    def get_importance(self, ) -> torch.Tensor:
        """
        获取特征的重要性得分
        """
        return torch.sigmoid(self.logits)

    def decay(self, ) -> None:
        """
        按照一定的比例衰减tau, keep, noise
        """
        self.tau = max(self.tau * self.tau_decay, self.min_tau)
        self.keep = max(self.keep * self.keep_decay, self.min_keep)
        self.noise = max(self.noise * self.noise_decay, self.min_noise)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == len(self.logits)

        if self.training:
            gumbels: torch.Tensor = -torch.empty_like(
                self.logits
            ).exponential_().log()
            gumbels = (self.logits + gumbels * self.noise) / self.tau
            mask_soft: torch.Tensor = gumbels.sigmoid()
        else:
            mask_soft = (self.logits / self.tau).sigmoid()

        q: torch.Tensor = torch.quantile(mask_soft, 1.0 - self.keep)
        mask_hard = (mask_soft >= q).float()
        mask: torch.Tensor = mask_hard - mask_soft.detach() + mask_soft
        mask = mask.view(*([1] * (x.dim() - 1)), -1)
        return x * mask, mask_soft


class GFSNetwork(CSModel):

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        stats_csv_file: str = "",
    ) -> None:
        super().__init__(
            x_fields=x_fields,
            y_fields=y_fields,
        )

        self.preprocessor = Preprocessor(
            x_fields=x_fields,
            return_list=["winsor_min_max", ],
            stats_csv_file=stats_csv_file,
        )
        self.gate = GumbelSigmoidGate(self.preprocessor.output_dim)

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

            nn.Linear(128, 1),
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
        x: torch.Tensor = self.preprocessor(data["x"])

        # 推理得到输出
        b, t, n, d = x.shape
        assert t == 1
        x, mask = self.gate(x)
        y_pred: torch.Tensor = self.linear(x.reshape(-1, d)).reshape(b, n, -1)

        # 对输出标准化
        mean: torch.Tensor = y_pred.mean(dim=-2, keepdim=True)
        std: torch.Tensor = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred, "mask": mask}

    def loss_func(self, engine: AccelerateEngine) -> Dict[str, torch.Tensor]:
        """
        使用ic loss作为默认损失函数
        """
        _ic_loss: torch.Tensor = ic_loss(
            engine.data["y"], engine.output["y_pred"], dim=-2
        )
        reg_loss: torch.Tensor = torch.relu(
            engine.output["mask"].mean() - self.gate.keep
        )

        return {
            "loss": _ic_loss + 0.2 * reg_loss,
            "ic_loss": _ic_loss,
            "reg_loss": reg_loss,
        }

    def on_epoch_end(self, engine: AccelerateEngine) -> None:
        self.gate.decay()
        engine.logger.info(
            f"tau: {self.gate.tau}, "
            f"keep: {self.gate.keep}, "
            f"noise: {self.gate.noise}"
        )

    def on_val_end(self, engine: AccelerateEngine) -> None:
        """
        储存每次验证结束后的重要度得分
        """
        importance_df = pd.DataFrame(
            {"gfs": self.gate.get_importance().detach().cpu().numpy()},
            index=self.x_fields
        ).sort_values("gfs", ascending=False)
        importance_df.to_csv(
            os.path.join(engine.folder, f"importance_{engine.current_step}.csv")
        )
