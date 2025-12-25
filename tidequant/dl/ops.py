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
    
    # 注释掉的写法由于*shape导致无法通过jit编译而弃用
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
    return -torch.nanmean(ic)


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


def tail_weight_ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int,
    tail_q: float = 0.2,
    tail_w: float = 3.0,
) -> torch.Tensor:
    """
    计算两个tensor之间尾部加权横截面ic

    dim表示进行ic计算的维度, 其他维度取平均

    自动忽略nan
    """

    assert y.shape == y_pred.shape

    mask = torch.isfinite(y) & torch.isfinite(y_pred)
    y0 = torch.nan_to_num(y, nan=0.0)
    p0 = torch.nan_to_num(y_pred, nan=0.0)

    # y 的“分位位置” qpos in [0,1]（用 rank 近似）
    y_fill = y0.masked_fill(~mask, float("inf"))
    sort_idx = torch.argsort(y_fill, dim=dim)
    ranks = torch.empty_like(sort_idx, dtype=y0.dtype)
    ar = torch.arange(y.size(dim), device=y.device, dtype=y0.dtype)
    view = [1] * y.ndim
    view[dim] = -1
    ranks.scatter_(dim, sort_idx, ar.view(view).expand_as(sort_idx))

    n_valid = mask.sum(dim=dim, keepdim=True)
    qpos = (ranks / (n_valid - 1).clamp_min(1)).masked_fill(~mask, float("nan"))

    # 权重：硬头尾 or 平滑头尾
    if smooth_gamma is None:
        w = torch.full_like(y0, mid_w)
        w = torch.where(qpos <= q, y0.new_tensor(tail_w), w)
        w = torch.where(qpos >= 1.0 - q, y0.new_tensor(head_w), w)
    else:
        dist = torch.nan_to_num((qpos - 0.5).abs() * 2.0, nan=0.0)  # 0(中间) -> 1(两端)
        ext = torch.where(qpos >= 0.5, y0.new_tensor(head_w), y0.new_tensor(tail_w))
        w = mid_w + (ext - mid_w) * dist.pow(float(smooth_gamma))

    w = w.masked_fill(~mask, 0.0)

    # 加权 Pearson IC
    wsum = torch.sum(w, dim=dim, keepdim=True)
    y_mean = torch.sum(y0 * w, dim=dim, keepdim=True) / (wsum + eps)
    p_mean = torch.sum(p0 * w, dim=dim, keepdim=True) / (wsum + eps)

    yc = y0 - y_mean
    pc = p0 - p_mean
    cov = torch.sum(w * yc * pc, dim=dim)
    y_var = torch.sum(w * yc * yc, dim=dim)
    p_var = torch.sum(w * pc * pc, dim=dim)

    ic = cov / (torch.sqrt(y_var + eps) * torch.sqrt(p_var + eps) + eps)
    ic = ic.masked_fill(n_valid.squeeze(dim) < 2, float("nan"))
    return -torch.nanmean(ic)
