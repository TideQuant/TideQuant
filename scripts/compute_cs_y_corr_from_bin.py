"""
基于bin截面数据库计算y之间的相关矩阵
"""

from typing import List

import jsonargparse
import numpy as np
import pandas as pd
import torch
import xarray as xr

from tidequant.data import BinCSDataBase
from tidequant.dl import corr


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_file", type=str, required=True)
    parser.add_argument("--start_dt", type=str, default=None)
    parser.add_argument("--end_dt", type=str, default=None)
    parser.add_argument("--start_second", type=int, default=None)
    parser.add_argument("--end_second", type=int, default=None)
    parser.add_argument("--step_second", type=int, default=None)
    args: jsonargparse.Namespace = parser.parse_args()

    from_db = BinCSDataBase(args.from_folder)
    y_fields: List[str] = from_db.list_y_fields()
    y: xr.DataArray = from_db.read_multi_data(
        mode='y',
        fields=y_fields,
        date_slice=slice(
            np.datetime64(args.start_dt), np.datetime64(args.end_dt)
        ),
        second_slice=slice(
            args.start_second, args.end_second, args.step_second
        ),
    )
    y_gpu: torch.Tensor = torch.as_tensor(
        y.transpose("field", ...).data,
        dtype=torch.float32,
        device="cuda"
    ).reshape(len(y_fields), -1)
    y_corr: np.ndarray = corr(y_gpu).detach().cpu().numpy()
    
    assert args.to_file.endswith(".csv")
    df = pd.DataFrame(y_corr, index=y_fields, columns=y_fields)
    df.to_csv(args.to_file)
