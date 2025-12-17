"""
通用运算函数, 损失函数和子模块等
"""

from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


"""
运算函数
"""

def rank(
    x: torch.Tensor,
    dim: int,
    bins: Optional[int] = None,
) -> torch.Tensor:
    """
    计算x在dim维度的排名并归一化到[0, 1]

    nan会被填充为0.5
    """

    mask: torch.Tensor = ~torch.isnan(x)
    # 将nan填充为inf, 使其不影响其他值的排序
    x = torch.where(mask, x, torch.full_like(x, float("inf")))

    # 计算rank
    _, order = torch.topk(x, k=x.size(dim), dim=dim, sorted=True, largest=False)
    rank: torch.Tensor = torch.empty_like(order)
    
    # shape: List[int] = [1] * x.ndim
    # shape[dim] = -1
    # idx = torch.arange(
    #     x.shape[dim], device=x.device
    # ).view(*shape).expand_as(order)
    # rank.scatter_(dim, order, idx)

    nd: int = x.dim()
    d: int = dim if dim >= 0 else dim + nd
    
    idx: torch.Tensor = torch.arange(
        x.size(d), device=x.device, dtype=order.dtype
    )
    for i in range(d):
        idx = idx.unsqueeze(0)
    
    for i in range(nd - d - 1):
        idx = idx.unsqueeze(-1)

    idx = idx.expand_as(order)
    rank.scatter_(d, order, idx)

    # 将rank归一化到0-1
    cnt: torch.Tensor = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    rank = (rank.float() + 0.5) / cnt.float()
    rank = torch.where(mask, rank, torch.full_like(rank, 0.5))

    # 分箱
    if bins is not None:
        bin_idx: torch.Tensor = torch.clamp(
            (rank * bins).floor(), min=0, max=bins - 1
        )
        rank = (bin_idx + 0.5) / bins
    return rank.clamp_(0.0, 1.0)


def nanmedian(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ONNX更易支持的nanmedian
    """

    nan_mask: torch.Tensor = torch.isnan(x)
    filled: torch.Tensor = torch.where(
        nan_mask, torch.full_like(x, float("inf")), x
    )
    sorted_vals, sorted_idx = torch.sort(filled, dim=dim)

    valid: torch.Tensor = (~nan_mask).sum(dim=dim, keepdim=True)
    kth: torch.Tensor = ((valid - 1) // 2).clamp_min(0).to(torch.long)
    vals: torch.Tensor = torch.gather(sorted_vals, dim, kth)
    vals = torch.where(valid > 0, vals, torch.full_like(vals, float("nan")))
    idxs: torch.Tensor = torch.gather(sorted_idx, dim, kth)
    idxs = torch.where(valid > 0, idxs, torch.zeros_like(idxs))
    
    if not keepdim:
        vals = vals.squeeze(dim)
        idxs = idxs.squeeze(dim)
    return vals, idxs


"""
损失函数
"""

def mae_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,    
) -> torch.Tensor:
    """
    计算两个tensor之间的L1损失函数

    自动忽略nan
    """

    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    return F.l1_loss(y[mask], y_pred[mask])


def mse_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,    
) -> torch.Tensor:
    """
    计算两个tensor之间的L2损失函数

    自动忽略nan
    """

    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    return F.mse_loss(y[mask], y_pred[mask])


def ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int,
    mean: bool = True,
) -> torch.Tensor:
    """
    计算两个tensor之间的横截面ic

    dim表示进行ic计算的维度, 其他维度取平均

    自动忽略nan
    """
    
    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)

    # 计算没有被mask掉的均值
    count: torch.Tensor = torch.sum(mask, dim=dim, keepdim=True)
    y_mean: torch.Tensor = torch.sum(
        (y * mask), dim=dim, keepdim=True
    ) / (count + 1e-8)
    y_pred_mean: torch.Tensor = torch.sum(
        (y_pred * mask), dim=dim, keepdim=True
    ) / (count + 1e-8)

    # 计算协方差和方差
    y_center: torch.Tensor = (y - y_mean) * mask
    y_pred_center: torch.Tensor = (y_pred - y_pred_mean) * mask
    cov: torch.Tensor = torch.sum(y_center * y_pred_center, dim=dim)
    y_var: torch.Tensor = torch.sum(y_center * y_center, dim=dim)
    y_pred_var: torch.Tensor = torch.sum(y_pred_center * y_pred_center, dim=dim)

    # 计算相关系数
    ic: torch.Tensor = cov / (torch.sqrt((y_var + 1e-8) * (y_pred_var + 1e-8)))
    if mean:
        return -torch.nanmean(ic)
    return -ic


def soft_rank(
    x: torch.Tensor,
    n_bins: int = 4800,
    tau: float = 1e-4,
) -> torch.Tensor:
    """
    通过softmax近似的可导rank
    """

    assert x.ndim == 1
    if x.numel() == 0:
        return x

    # 计算等频分箱分位数点
    quantiles: torch.Tensor = torch.linspace(
        0, 1, n_bins + 1, device=x.device
    )[1: -1]
    bin_edges: torch.Tensor = torch.quantile(x.detach(), quantiles)

    # 添加最小值和最大值
    min_val: torch.Tensor = x.min()
    max_val: torch.Tensor = x.max()
    bin_edges: torch.Tensor = torch.cat(
        [min_val.unsqueeze(0), bin_edges, max_val.unsqueeze(0)]
    )
    
    # 计算每个点到所有边界的距离
    diffs: torch.Tensor = x.unsqueeze(1) - bin_edges.unsqueeze(0)
    
    # 使用softmax计算每个点属于每个区间的概率
    probs: torch.Tensor = torch.softmax(-diffs.abs() / tau, dim=1)

    # 计算每个点的软秩次
    bin_indices: torch.Tensor = torch.arange(
        n_bins + 1, device=x.device
    ).float()
    soft_ranks: torch.Tensor = (probs * bin_indices).sum(dim=1)

    # 归一化到[0, 1]范围
    return soft_ranks / n_bins


def rank_ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int,
    n_bins: int = 4800,
    tau: float = 1e-4,
) -> torch.Tensor:
    """
    计算两个tensor之间的近似横截面rank ic损失函数

    dim表示进行rank ic计算的维度, 其他维度取平均

    自动忽略nan
    """
    assert y.shape == y_pred.shape

    dim = dim if dim >= 0 else dim + y.ndim
    y = y.movedim(dim, -1)
    y_pred = y_pred.movedim(dim, -1)

    y = y.reshape(-1, y.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    loss: torch.Tensor = y.new_tensor(0.0)
    mask: torch.Tensor = ~torch.isnan(y)

    for i in range(y.shape[0]):
        _y: torch.Tensor = y[i][mask[i]]
        _y_pred: torch.Tensor = y_pred[i][mask[i]]

        _y = soft_rank(_y, n_bins=n_bins, tau=tau)
        _y_pred = soft_rank(_y_pred, n_bins=n_bins, tau=tau)
        loss += ic_loss(_y, _y_pred, dim=-1)

    return loss / y.shape[0]


def skew_loss(
    x: torch.Tensor,
    dim: int,
    target: float = 0.0,
    p: int = 1,
) -> torch.Tensor:
    """
    计算在dim维度上的偏度损失, 默认通过mean聚合
    """

    mean: torch.Tensor = x.mean(dim=dim, keepdim=True)
    diff: torch.Tensor = x - mean
    var: torch.Tensor = (diff ** 2).mean(dim=dim, keepdim=True)
    std: torch.Tensor = (var + 1e-8).sqrt()
    skew = (diff / std).pow(3).mean(dim=dim)
    return (skew - target).abs().pow(p).mean()
