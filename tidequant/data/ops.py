"""
自定义运算符
"""

import numpy as np
import xarray as xr
from scipy.stats import spearmanr


def ic(
    x1: np.ndarray | xr.DataArray,
    x2: np.ndarray | xr.DataArray,
    axis: int,
    rank: bool = False,
) -> float:
    """
    计算两个多维矩阵的IC/Rank IC
    
    axis表示进行IC计算的维度, 其他维度取平均

    自动忽略nan
    """

    if isinstance(x1, xr.DataArray):
        x1 = x1.data
    
    if isinstance(x2, xr.DataArray):
        x2 = x2.data

    assert x1.shape == x2.shape

    axis = axis if axis >= 0 else x1.ndim + axis
    x1: np.ndarray = np.moveaxis(x1, axis, -1)
    x2: np.ndarray = np.moveaxis(x2, axis, -1)

    ics: np.ndarray = np.zeros(x1.shape[: -1], dtype=np.float32)
    for idx in np.ndindex(x1.shape[: -1]):
        ics[idx] = _ic_1d(x1[idx], x2[idx], rank=rank)
    return np.nanmean(ics)


def _ic_1d(x1: np.ndarray, x2: np.ndarray, rank: bool = False) -> float:
    """
    计算两个一维矩阵的IC/Rank IC
    
    自动忽略nan
    """

    assert x1.ndim == 1
    assert x2.shape == x1.shape

    mask: np.ndarray = np.isfinite(x1) & np.isfinite(x2)

    if rank:
        rank_ic, _ = spearmanr(x1[mask], x2[mask])
        return rank_ic
    return np.corrcoef(x1[mask], x2[mask])[0, 1]


def zscore(x: np.ndarray | xr.DataArray, axis: int) -> np.ndarray:
    """
    对x按照某一维度做zscore

    自动忽略nan
    """

    is_xarray: bool = isinstance(x, xr.DataArray)
    if is_xarray:
        _x = x.data
    else:
        _x = x

    mean: np.ndarray = np.nanmean(_x, axis=axis, keepdims=True)
    std: np.ndarray = np.nanstd(_x, axis=axis, keepdims=True)
    x_zscore: np.ndarray = (_x - mean) / std

    if is_xarray:
        return x.copy(data=x_zscore)
    return x_zscore
