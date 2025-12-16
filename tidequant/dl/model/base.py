"""
模型基类
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import torch
import torch.onnx as onnx
import xarray as xr
from torch import nn
from torch.utils.data import Dataset

from ..callback import Callback
from ..ops import ic_loss, rank_ic_loss
from ...data.ops import ic

if TYPE_CHECKING:
    from ..engine import AccelerateEngine


class Model(nn.Module, Callback, ABC):
    """
    模型基类
    """

    def get_optimizer(self, engine: AccelerateEngine) -> torch.optim.Optimizer:
        """
        初始化优化器

        默认使用AdamW优化器
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=engine.params["lr"],
            weight_decay=engine.params["weight_decay"],
        )

    @abstractmethod
    def loss_func(self, engine: AccelerateEngine) -> Dict[str, torch.Tensor]:
        """
        计算loss字典, 其中loss键值为梯度下降所需的loss

        engine中的相关变量是engine.data, engine.output等
        """
        pass

    @abstractmethod
    def metric_func(
        self,
        engine: AccelerateEngine,
        dataset: Dataset,
    ) -> Dict[str, float]:
        """
        计算metric字典

        engine中的相关变量是engine.whole_output等
        """
        pass

    def export_jit(self, engine: AccelerateEngine) -> None:
        """
        导出为JIT模型并保存
        """
        raise NotImplementedError

    def export_onnx(self, engine: AccelerateEngine) -> None:
        """
        导出为ONNX模型并保存
        """
        raise NotImplementedError


class CSModel(Model):
    """
    横截面模型基类

    默认dataset中包含x_fields, y_fields和y:
    - y为(n_date, n_second, n_ticker, n_field)的xr.DataArray

    推荐在__init__函数只传入python的基本类型
    因为他们会在script脚本中使用jsonargparse自动添加
    """

    def __init__(
        self,
        x_fields: List[str],
        y_fields: List[str],
        seq_len: int = 1,
        use_mix_loss: bool = False,
    ) -> None:
        super().__init__()
        
        self.x_fields: List[str] = x_fields
        self.y_fields: List[str] = y_fields
        self.dim: int = len(x_fields)
        self.output_dim: int = len(y_fields)
        self.seq_len: int = seq_len
        self.use_mix_loss: bool = use_mix_loss

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        默认模型的输入data中包含x和y键值:
        - x的形状为(b, seq_len, n_ticker, n_field)
        - y的形状为(b, n_ticker, n_field)

        默认模型的输出output中包含y_pred键值:
        - y_pred的形状为(b, n_ticker, n_field)
        """
        x: torch.Tensor = data["x"]
        return {"y_pred": None}

    def loss_func(self, engine: AccelerateEngine) -> Dict[str, torch.Tensor]:
        """
        使用ic loss作为默认损失函数
        """
        _ic_loss: torch.Tensor = ic_loss(
            engine.data["y"], engine.output["y_pred"], dim=-2
        )
        if not self.use_mix_loss:
            return {"loss": _ic_loss}

        _rank_ic_loss: torch.Tensor = rank_ic_loss(
            engine.data["y"], engine.output["y_pred"], dim=-2
        )
        return {
            "loss": (_ic_loss + _rank_ic_loss) / 2.0,
            "ic_loss": _ic_loss,
            "rank_ic_loss": _rank_ic_loss,
        }

    def metric_func(
        self,
        engine: AccelerateEngine,
        dataset: Dataset,
    ) -> Dict[str, float]:
        """
        使用ic和rank ic作为默认评估指标

        只关注第一个标签的ic
        """
        y: np.ndarray = dataset.y[..., 0].data
        y_pred: np.ndarray = engine.whole_output[
            "y_pred"
        ].numpy()[..., 0].reshape(y.shape)
        
        _ic: float = ic(y, y_pred, axis=-1)
        _rank_ic: float = ic(y, y_pred, axis=-1, rank=True)
        return {
            "ic": _ic,
            "rank_ic": _rank_ic,
            "avg_ic": (_ic + _rank_ic) / 2.0,
        }

    def save_test_y(self, engine: AccelerateEngine) -> None:
        """
        保存y和y_pred到本地
        """
        y = engine.test_dataset.y.isel(field=0)
        y_pred = xr.DataArray(
            engine.whole_output["y_pred"].numpy()[..., 0].reshape(y.shape),
            coords=y.coords,
            dims=y.dims,
            name="y_pred",
        )
        xr.Dataset({"y": y, "y_pred": y_pred}).to_netcdf(
            os.path.join(engine.folder, "y.nc")
        )

    def export_jit(self, engine: AccelerateEngine) -> None:
        """
        导出为JIT模型并保存
        """
        class _ExportWrap(nn.Module):
            """
            输入(b, seq_len, n_ticker, n_field)的x
            
            输出(b, n_ticker)的y_pred
            """

            def __init__(self, model: "CSModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out: Dict[str, torch.Tensor] = self.model({"x": x})
                return out["y_pred"][:, :, 0]

        scripted = torch.jit.script(_ExportWrap(self.eval()))
        scripted.save(os.path.join(engine.folder, "model_jit.pt"))

    def export_onnx(self, engine: AccelerateEngine) -> None:
        """
        导出为股票维度可动态变化的ONNX模型并保存

        实盘时固定batch_size为1
        """
        class _ExportWrap(nn.Module):
            """
            输入(seq_len, n_ticker, n_field)的x

            输出(1, n_ticker)的y_pred
            """

            def __init__(self, model: "CSModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model({"x": x.unsqueeze(0)})["y_pred"][:, :, 0]

        onnx.export(
            _ExportWrap(self.eval()),
            torch.randn(self.seq_len, 5736, self.dim),
            os.path.join(engine.folder, "model.onnx"),
            verbose=True,
            opset_version=17,
            input_names=["x"],
            output_names=["y_pred"],
            dynamic_axes={
                "x": {1: "ticker"},
                "y_pred": {1: "ticker"},
            },
        )
