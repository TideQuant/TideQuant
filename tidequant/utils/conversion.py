"""
数值转换相关的辅助工具
"""

import numpy as np


def validate_float_to_int(value: float) -> int:
    """
    将浮点数转换为整数

    如果转换过程中存在精度损失, 则抛出AssertionError
    """

    assert value == int(value), f"lossy in int({value})"
    return int(value)


def validate_dt_astype(
    dt: np.datetime64,
    dtype: str,
) -> np.datetime64:
    """
    将np.datetime64转换为另一种精度

    如果转换过程中存在精度损失, 则抛出AssertionError
    """

    new_dt = dt.astype(dtype)
    assert new_dt.astype(dt.dtype) == dt, f"lossy in convert {dt} to {dtype}"
    return new_dt
