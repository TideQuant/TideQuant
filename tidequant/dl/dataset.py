"""
数据集类
"""

from typing import Dict, List

import numpy as np
import xarray as xr
from torch.utils.data import Dataset

from tidequant.data.cs_database import BinCSDataBase, HDF5CSDataBase


class HDF5CSDataset(Dataset):
    """
    基于分日h5数据库的数据集

    数据库格式参考data/cs_database/HDF5CSDataBase

    传入的切片均为双闭区间
    """

    def __init__(
        self,
        bin_folder: str,
        h5_folder: str,
        y_fields: str | List[str],
        date_slice: slice = slice(None),
        second_slice: slice = slice(None),
        seq_len: int = 1,
    ) -> None:
        self.bin_db = BinCSDataBase(bin_folder)
        self.h5_db = HDF5CSDataBase(h5_folder, bin_folder)
        self.date_slice: slice = date_slice
        self.second_slice: slice = second_slice
        self.tickers = slice(None)
        self.x_fields: List[str] = self.h5_db.x_fields
        self.y_fields: List[str] = [y_fields, ] if isinstance(
            y_fields, str
        ) else y_fields
        self.seq_len: int = seq_len

        self.y: xr.DataArray = self.bin_db.read_multi_data(
            mode="y",
            fields=self.y_fields,
            date_slice=self.date_slice,
            second_slice=self.second_slice,
            tickers=self.tickers,
            return_raw=False,
            n_worker=0,
        )
        self.time_idxes: List[int] = self.h5_db.select_time_idxes(
            date_slice=self.date_slice,
            second_slice=self.second_slice,
            seq_len=self.seq_len
        )
        assert len(self.time_idxes) == len(self.y.date) * len(self.y.second)

    def __len__(self) -> int:
        return len(self.time_idxes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        time_idx: int = self.time_idxes[idx]
        x_list: List[np.ndarray] = [
            self.h5_db.get_x_with_time_idx(time_idx - (self.seq_len - 1 - i))
            for i in range(self.seq_len)
        ]
        x: np.ndarray = np.stack(x_list, axis=0)

        date_idx: int = idx // len(self.y.second)
        second_idx: int = idx % len(self.y.second)
        y: np.ndarray = self.y.data[date_idx, second_idx]
        return {
            "x": x,
            "y": y,
            "idx": np.array([idx, ], dtype=np.int64),
        }
