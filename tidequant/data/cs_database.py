"""
用于读取截面数据的数据库类
"""

import os
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from numpy.typing import ArrayLike
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..utils.io import read_txt_lines


def _convert_slice_to_xr_slice(index: ArrayLike, slc: slice) -> slice:
    """
    把Python左闭右开slice转成xarray可用的左闭右闭slice
    """

    if slc.stop is None:
        return slc

    pos = index.searchsorted(slc.stop, side="left") - 1
    assert pos >= 0
    return slice(slc.start, index[pos], slc.step)


def slice_from_cs_da(
    data: xr.DataArray,
    date_slice: slice = slice(None),
    second_slice: slice = slice(None),
    tickers: List[str] | slice = slice(None),
    fields: List[str] | slice = slice(None),
) -> xr.DataArray:
    """
    从截面数据中按日期/秒/品种/字段切片, 支持步长采样
    """

    # 转化为xarray可用的左闭右闭slice
    date_slice = _convert_slice_to_xr_slice(data.date, date_slice)
    second_slice = _convert_slice_to_xr_slice(data.second, second_slice)

    sliced_data: xr.DataArray = data.sel(
        date=slice(date_slice.start, date_slice.stop)
        if date_slice != slice(None) else slice(None),
        second=slice(second_slice.start, second_slice.stop)
        if second_slice != slice(None) else slice(None),
        ticker=tickers,
        field=fields,
    )

    if date_slice.step not in (None, 1):
        sliced_data = sliced_data.isel(date=slice(None, None, date_slice.step))

    if second_slice.step not in (None, 1):
        sliced_data = sliced_data.isel(
            second=slice(None, None, second_slice.step)
        )
    return sliced_data


class BinCSDataBase:
    """
    基于bin文件构建的截面数据库

    数据库格式:
    datetime.bin: UTC时区的ns时间戳
    ticker.bin: 品种名
    x/*: 一个bin文件对应一个因子
        (n_date, n_second, n_ticker)
        第二维表示日内的秒级间隔
    y/*: 一个bin文件对应一个标签
        (n_date, n_second, n_ticker)

    mask_name: 控制可交易mask的文件名, 为空则不应用mask
    tz_offset: 控制时区偏移小时数
    """

    def __init__(
        self,
        folder: str,
        mask_name: str = "TradableUniv",
        tz_offset: int = 8,
    ) -> None:
        self.folder: str = folder
        self.mask_name: str = mask_name
        
        self._load_dts(tz_offset)
        
        self.tickers: np.ndarray = np.loadtxt(
            os.path.join(self.folder, "ticker.bin"),
            dtype=str,
        )

    def _load_dts(self, tz_offset: int) -> None:
        """
        加载时间并按目标时区偏移, 生成日期与日内距离开盘的秒坐标
        """
        timestamps: np.ndarray = np.fromfile(
            os.path.join(self.folder, "datetime.bin"),
            dtype=np.int64,
        ) // (10 ** 9)
        self.dts: np.ndarray = (
            timestamps.astype("datetime64[s]")
            + np.timedelta64(tz_offset, "h")
        )
        
        dates: np.ndarray = self.dts.astype("datetime64[D]")
        self.dates: np.ndarray = np.unique(dates)
        
        first_day_mask: np.ndarray = dates == self.dates[0]
        seconds: np.ndarray = (
            (self.dts[first_day_mask] - self.dts[first_day_mask][0])
            / np.timedelta64(1, "s")
        )
        self.seconds: np.ndarray = seconds.astype(int)

    def list_x_fields(self) -> List[str]:
        """
        返回所有因子名称
        """
        return os.listdir(os.path.join(self.folder, "x"))

    def list_y_fields(self) -> List[str]:
        """
        返回所有标签名称
        """
        return os.listdir(os.path.join(self.folder, "y"))

    def _read_data(
        self,
        file: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> xr.DataArray:
        """
        读取bin文件并返回DataArray

        日期为[start_dt, end_dt)
        """
        start: int = 0
        if start_dt is not None:
            start = np.searchsorted(self.dates, start_dt, side="left")
        
        end: int = len(self.dates)
        if end_dt is not None:
            end = np.searchsorted(self.dates, end_dt, side="left")
        
        data: np.ndarray = np.fromfile(
            file,
            dtype=np.float32,
            offset=(
                start * len(self.seconds) * len(self.tickers)
                * np.dtype(np.float32).itemsize
            ),
            count=(end - start) * len(self.seconds) * len(self.tickers),
        )
        data = data.reshape(
            end - start,
            len(self.seconds),
            len(self.tickers),
            1,
        )
        
        field: str = os.path.basename(file)
        return xr.DataArray(
            data,
            coords={
                "date": self.dates[start:end],
                "second": self.seconds,
                "ticker": self.tickers,
                "field": [field],
            },
            dims=("date", "second", "ticker", "field"),
            name=field,
        )

    def read_x(
        self,
        field: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> xr.DataArray:
        """
        读取单个因子
        """
        return self._read_data(
            file=os.path.join(self.folder, "x", field),
            start_dt=start_dt,
            end_dt=end_dt,
        )

    def read_y(
        self,
        field: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        return_raw: bool = False,
    ) -> xr.DataArray:
        """
        读取单个标签, 可选择是否处理inf和应用mask
        """
        y: xr.DataArray = self._read_data(
            file=os.path.join(self.folder, "y", field),
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if return_raw:
            return y
        
        # 将inf替换为nan
        y = y.where(~np.isinf(y))
        
        # 应用mask
        if self.mask_name:
            mask: np.ndarray = self._read_data(
                file=os.path.join(self.folder, "x", self.mask_name),
                start_dt=start_dt,
                end_dt=end_dt,
            ).data.astype(bool)
            y.data[~mask] = np.nan
        return y

    def read_multi_data(
        self,
        mode: Literal['x', 'y'],
        fields: str | List[str],
        date_slice: slice = slice(None),
        second_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        return_raw: bool = False,
        n_worker: int = 32,
    ) -> xr.DataArray:
        """
        多线程读取多个数据并合并返回DataArray
        """
        if mode == "x":
            read_func: Callable[..., xr.DataArray] = self.read_x
        elif mode == "y":
            read_func = partial(self.read_y, return_raw=return_raw)
        else:
            raise ValueError(f"{mode} must be x or y")

        if isinstance(fields, str):
            fields = [fields]

        # 先读取第一个数据用于推断形状和开辟内存
        first_arr: xr.DataArray = slice_from_cs_da(
            data=read_func(fields[0], date_slice.start, date_slice.stop),
            date_slice=slice(None, None, date_slice.step),
            second_slice=second_slice,
            tickers=tickers,
        )
        first_values: np.ndarray = np.squeeze(first_arr.data, axis=-1)

        # 预先开辟目标矩阵, 后续多线程直接填充
        merged: np.ndarray = np.empty(
            (*first_values.shape, len(fields)), dtype=first_values.dtype
        )
        merged[:, :, :, 0] = first_values

        def read_and_assign(i: int) -> None:
            arr: xr.DataArray = slice_from_cs_da(
                data=read_func(
                    fields[i], date_slice.start, date_slice.stop
                ),
                date_slice=slice(None, None, date_slice.step),
                second_slice=second_slice,
                tickers=tickers,
            )
            merged[:, :, :, i] = np.squeeze(arr.data, axis=-1)

        if n_worker == 0 or len(fields) == 1:
            # 退化到单线程
            for i in range(1, len(fields)):
                read_and_assign(i)
        else:
            worker_num: int = min(n_worker, len(fields) - 1)
            with ThreadPoolExecutor(worker_num) as executor:
                futures = [
                    executor.submit(read_and_assign, i)
                    for i in range(1, len(fields))
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures)
                ):
                    future.result()

        fields: np.ndarray = np.array(fields, dtype=object)
        return xr.DataArray(
            merged,
            coords={
                "date": first_arr.date.data,
                "second": first_arr.second.data,
                "ticker": first_arr.ticker.data,
                "field": fields,
            },
            dims=("date", "second", "ticker", "field"),
            name=mode,
        )

    def read_dataset(
        self,
        x_fields: str | List[str],
        y_fields: str | List[str],
        date_slice: slice = slice(None),
        second_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        n_worker: int = 32,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        读取数据集, 返回x和y的DataArray
        """
        kwargs: Dict[str, Any] = {
            "date_slice": date_slice,
            "second_slice": second_slice,
            "tickers": tickers,
            "n_worker": n_worker,
        }

        if isinstance(x_fields, str):
            x_fields = [x_fields]

        x: xr.DataArray = self.read_multi_data(
            mode='x',
            fields=x_fields,
            **kwargs,
        )

        if isinstance(y_fields, str):
            y_fields = [y_fields, ]

        y: xr.DataArray = self.read_multi_data(
            mode='y',
            fields=y_fields,
            **kwargs,
        )
        return x, y

    def read_flatten_dataset(
        self,
        x_fields: str | List[str],
        y_field: str,
        date_slice: slice = slice(None),
        second_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        apply_func_x: Callable[[xr.DataArray], xr.DataArray] = lambda x: x,
        apply_func_y: Callable[[xr.DataArray], xr.DataArray] = lambda x: x,
        n_worker: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取按y展平后的数据集, 常用于训练树
        """
        # 读取y, 预处理后展平
        y: xr.DataArray = self.read_y(
            y_field, date_slice.start, date_slice.stop
        )
        y = slice_from_cs_da(
            data=y,
            date_slice=slice(None, None, date_slice.step),
            second_slice=second_slice,
            tickers=tickers,
        )
        y = apply_func_y(y)
        mask: np.ndarray = ~np.isnan(y.data)
        flatten_y: np.ndarray = y.data[mask]

        if isinstance(x_fields, str):
            x_fields = [x_fields, ]

        flatten_x: np.ndarray = np.empty(
            (len(flatten_y), len(x_fields)), dtype=np.float32
        )

        def read_and_assign(i: int) -> None:
            x_da: xr.DataArray = self.read_x(
                x_fields[i], date_slice.start, date_slice.stop
            )
            x_da = slice_from_cs_da(
                data=x_da,
                date_slice=slice(None, None, date_slice.step),
                second_slice=second_slice,
                tickers=tickers,
            )
            x_da = apply_func_x(x_da)
            flatten_x[:, i] = x_da.data[mask]

        if n_worker == 0 or len(x_fields) == 1:
            for i in range(len(x_fields)):
                read_and_assign(i)
        else:
            with ThreadPoolExecutor(
                min(n_worker, len(x_fields))
            ) as executor:
                futures = [
                    executor.submit(read_and_assign, i)
                    for i in range(len(x_fields))
                ]
                for future in tqdm(
                    as_completed(futures), total=len(x_fields)
                ):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error {e}")
        return flatten_x, flatten_y


class HDF5CSDataBase:
    """
    基于h5文件构建的截面数据库
    
    每个h5文件对应一个日期的固定因子数据, 数据库格式如下:
    x_list.txt: 因子名
    {date}.h5:
        "x": 因子矩阵(1, n_second, n_ticker, n_x_field)

    数据库的内存分布为同一时刻因子连续, 适合深度学习训练
    """

    def __init__(self, folder: str, bin_folder: str) -> None:
        self.folder: str = folder
        self.bin_db = BinCSDataBase(bin_folder)

        # 获取数据的元信息
        files: List[str] = [
            file for file in os.listdir(self.folder) if file[-3: ] == ".h5"
        ]
        files.sort()
        self.dates = np.array(
            [file.split('.')[0] for file in files],
            dtype="datetime64[D]"
        )
        self.files: List[str] = [
            os.path.join(self.folder, file) for file in files
        ]
        self.seconds: np.ndarray = self.bin_db.seconds
        self.tickers: np.ndarray = self.bin_db.tickers
        self.x_fields: List[str] = read_txt_lines(
            os.path.join(self.folder, "x_list.txt")
        )

        # 读取一个文件检查shape是否正确
        with h5py.File(self.files[0], "r") as f:
            x_shape: Tuple[int, ...] = f["x"].shape
            assert x_shape == (
                1, len(self.seconds), len(self.tickers), len(self.x_fields),
            )

        # 维护一个句柄池, 在读取到相应文件时再打开句柄
        self.handle_pool = H5HandlePool(self.files)
        self._set_x_slices: bool = False

    def get_x_with_time_idx(self, idx: int) -> np.ndarray:
        """
        根据时间idx读取相应的截面数据
        """
        assert 0 <= idx < len(self.dates) * len(self.seconds)
        date_idx: int = idx // len(self.seconds)
        second_idx: int = idx % len(self.seconds)
        return self.handle_pool.get(date_idx)["x"][0, second_idx, :, :]

    def select_time_idxes(
        self,
        date_slice: slice = slice(None),
        second_slice: slice = slice(None),
        seq_len: int = 1,
    ) -> List[int]:
        """
        根据日期和秒索引获取所有可用的时间idx

        如果seq_len取不到也需要排除在外, 连续性按全局时间idx判断
        """
        time_grid = xr.DataArray(
            np.arange(len(self.dates) * len(self.seconds)).reshape(
                len(self.dates), len(self.seconds)
            ),
            coords={"date": self.dates, "second": self.seconds},
            dims=["date", "second"],
        )
        date_slice_idx: slice = time_grid.get_index("date").slice_indexer(
            date_slice.start, date_slice.stop, date_slice.step
        )
        second_slice_idx: slice = time_grid.get_index("second").slice_indexer(
            second_slice.start, second_slice.stop, second_slice.step
        )
        time_idxes: List[int] = time_grid.data[
            date_slice_idx, second_slice_idx
        ].ravel().tolist()
        return [idx for idx in time_idxes if idx >= seq_len - 1]


class H5HandlePool:
    """
    固定大小的h5句柄池, 简单LRU
    """

    def __init__(self, files: List[str], max_n_handle: int = 2048) -> None:
        self.files: List[str] = files
        self.max_n_handle: int = max_n_handle
        
        self.pool: Dict[int, h5py.File] = {}
        self.order: List[int] = []

    def get(self, idx: int) -> h5py.File:
        """
        获取idx对应的句柄
        """
        if idx in self.pool:
            self.order.remove(idx)
            self.order.append(idx)
            return self.pool[idx]

        handle = h5py.File(self.files[idx], "r", locking=False)
        self.pool[idx] = handle
        self.order.append(idx)
 
        if len(self.order) > self.max_n_handle:
            # 移除最早的句柄
            old_idx: int = self.order.pop(0)
            self.pool.pop(old_idx).close()
        return handle
