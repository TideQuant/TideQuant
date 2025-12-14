# coding: utf-8
"""
基于bin截面数据库计算因子统计指标
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from tidequant.data import BinCSDataBase, slice_from_cs_da
from tidequant.utils import read_txt_lines


def _read_and_compute_stats(
    from_folder: str,
    field: str,
    date_slice: slice = slice(None),
    second_slice: slice = slice(None),
    tickers: List[str] | slice = slice(None),
) -> Dict[str, float]:
    """
    读取单个因子并计算分位数
    """

    from_db = BinCSDataBase(from_folder)
    x: xr.DataArray = slice_from_cs_da(
        data=from_db.read_x(field, date_slice.start, date_slice.stop),
        date_slice=slice(None, None, date_slice.step),
        second_slice=second_slice,
        tickers=tickers,
    )

    x_stats: Dict[str, float] = {}
    (
        x_stats["x_1"],
        x_stats["x_25"],
        x_stats["x_50"],
        x_stats["x_75"],
        x_stats["x_99"],
    ) = np.nanpercentile(
        x.data, [1, 25, 50, 75, 99]
    )
    return x_stats


def build_norm_stats_db(
    from_folder: str,
    to_file: str,
    fields: List[str] | None,
    date_slice: slice = slice(None),
    second_slice: slice = slice(None),
    tickers: List[str] | slice = slice(None),
    n_worker: int = 32,
) -> None:
    """
    批量计算分位数并写入 CSV
    """

    from_db = BinCSDataBase(from_folder)
    if fields is None:
        fields = from_db.list_x_fields()

    with ProcessPoolExecutor(min(n_worker, len(fields))) as executor:
        futures = {
            executor.submit(
                _read_and_compute_stats,
                from_folder,
                field,
                date_slice,
                second_slice,
                tickers,
            ): i
            for i, field in enumerate(fields)
        }

        stats_list: List[Dict[str, float]] = [None for _ in range(len(fields))]
        for future in tqdm(as_completed(futures), total=len(fields)):
            try:
                stats_list[futures[future]] = future.result()
            except Exception as e:
                raise ValueError(f"{fields[futures[future]]} {e}")

    stats_df = pd.DataFrame(stats_list, index=fields, dtype=np.float32)
    stats_df.to_csv(to_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_file", type=str, required=True)
    parser.add_argument("--x_fields_file", type=str, default=None)
    parser.add_argument("--start_dt", type=str, default=None)
    parser.add_argument("--end_dt", type=str, default=None)
    parser.add_argument("--end_second", type=int, default=None)
    parser.add_argument("--n_worker", type=int, default=128)
    args = parser.parse_args()

    build_norm_stats_db(
        from_folder=args.from_folder,
        to_file=args.to_file,
        fields=None if args.x_fields_file is None else read_txt_lines(
            args.x_fields_file
        ),
        date_slice=slice(
            None if args.start_dt is None else np.datetime64(args.start_dt),
            None if args.end_dt is None else np.datetime64(args.end_dt),
        ),
        second_slice=slice(None, args.end_second),
        n_worker=args.n_worker,
    )
