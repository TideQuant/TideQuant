"""
基于bin截面数据库生成分日HDF5数据库
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import h5py
import jsonargparse
import numpy as np
from tqdm import tqdm

from tidequant.data import BinCSDataBase
from tidequant.utils.io import read_txt_lines, write_txt_lines


def _read_and_save_hdf5(
    from_folder: str,
    to_folder: str,
    fields: List[str],
    date: np.datetime64,
) -> None:
    """
    读取一天的数据并写入h5文件
    """

    from_db = BinCSDataBase(from_folder)
    x = from_db.read_multi_data(
        mode="x",
        fields=fields,
        date_slice=slice(date, date + np.timedelta64(1, 'D')),
        n_worker=0,
    )

    file_path = os.path.join(
        to_folder, f"{np.datetime_as_string(date, unit='D')}.h5"
    )
    with h5py.File(file_path, "w") as f:
        f.create_dataset("x", data=x.data)


def build_hdf5_db(
    from_folder: str,
    to_folder: str,
    fields: List[str],
    n_worker: int = 32,
) -> None:
    """
    基于 bin 数据库按指定因子名新建 HDF5 截面数据库
    """

    os.makedirs(to_folder, exist_ok=False)
    write_txt_lines(os.path.join(to_folder, "x_list.txt"), fields)

    with ProcessPoolExecutor(max(1, n_worker)) as executor:
        print("start read and save date.h5")
        from_db = BinCSDataBase(from_folder)

        futures = [
            executor.submit(
                _read_and_save_hdf5,
                from_folder,
                to_folder,
                fields,
                date,
            )
            for date in from_db.dates
        ]

        for future in tqdm(as_completed(futures), total=len(from_db.dates)):
            try:
                future.result()
            except Exception as e:  # pylint: disable=broad-except
                print(e)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_folder", type=str, required=True)
    parser.add_argument("--x_fields_file", type=str, required=True)
    parser.add_argument("--n_worker", type=int, default=32)
    args: jsonargparse.Namespace = parser.parse_args()

    build_hdf5_db(
        from_folder=args.from_folder,
        to_folder=args.to_folder,
        fields=read_txt_lines(args.x_fields_file),
        n_worker=args.n_worker,
    )
