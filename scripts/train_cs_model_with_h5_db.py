"""
训练截面模型的脚本

支持预训练

支持训练确定参数后再用训练集+验证集训练一遍
"""

import inspect
import json
import jsonargparse
import os
import sys
from jsonargparse._util import import_object
from typing import Any, Dict, List, Type

import numpy as np
from setproctitle import setproctitle

from tidequant.dl import (
    AccelerateEngine,
    Callback,
    EarlyStopSaver,
    HDF5CSDataset,
    WarmUpSchedule,
    get_ckpt,
)
from tidequant.dl.model import CSModel
from tidequant.utils import validate_float_to_int


setproctitle("cross sectional model training")


def get_args() -> jsonargparse.Namespace:
    """
    从命令行获取参数
    """

    parser = jsonargparse.ArgumentParser()
    
    # 配置运行参数
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument(
        "--train_all_seed", type=int, nargs='+', default=[42, ]
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        nargs='+',
        default=["save_test_y", "export_onnx", "export_jit"],
    )

    # 配置AccelerateEngine
    parser.add_class_arguments(
        AccelerateEngine,
        "engine",
        skip={"callbacks", },
        instantiate=False,
    )

    # 配置Callbacks
    parser.add_argument("--metric_name", type=str, default="ic")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_n_val", type=int, default=sys.maxsize)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # 配置Dataset
    parser.add_argument("--bin_folder", type=str, required=True)
    parser.add_argument("--h5_folder", type=str, required=True)
    parser.add_argument("--y_fields", type=str, nargs='+', required=True)
    # 这里默认是Bar1m的配置, 均为左闭右开区间[start, end)
    parser.add_argument("--train_start_dt", type=str, default="2020-02-01")
    parser.add_argument("--train_end_dt", type=str, default="2024-11-01")
    parser.add_argument("--val_start_dt", type=str, default="2024-11-01")
    parser.add_argument("--val_end_dt", type=str, default="2025-01-01")
    parser.add_argument("--test_start_dt", type=str, default="2024-11-01")
    parser.add_argument("--test_end_dt", type=str, default="2025-01-01")
    parser.add_argument("--start_second", type=int, default=None)
    parser.add_argument("--end_second", type=int, default=None)
    parser.add_argument("--step_second", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=1)

    # 配置模型
    parser.add_subclass_arguments(
        CSModel,
        "model",
        skip={"x_fields", "y_fields", "seq_len"},
        instantiate=False,
    )
    return parser.parse_args()


def run_once(args: jsonargparse.Namespace) -> None:
    """
    执行一次训练
    """

    # 初始化引擎
    callbacks: List[Callback] = [EarlyStopSaver(
        metric_name=args.metric_name,
        patience=args.patience,
        max_n_val=args.max_n_val
    )]
    if args.warmup_ratio > 0:
        callbacks.append(WarmUpSchedule(args.warmup_ratio))

    engine = AccelerateEngine(
        folder=args.engine.folder,
        create_folder=args.engine.create_folder,
        callbacks=callbacks,
        params=args.engine.params,
    )

    # 保存参数
    if engine.accelerator.is_main_process:
        json.dump(
            vars(args),
            open(os.path.join(
                args.engine.folder, "args.json"
            ), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # 加载数据集
    dataset_kwargs: Dict[str, Any] = {
        "bin_folder": args.bin_folder,
        "h5_folder": args.h5_folder,
        "y_fields": args.y_fields,
        "seq_len": args.seq_len,
    }
    engine.set_dataset(
        train_dataset=HDF5CSDataset(
            date_slice=slice(
                np.datetime64(args.train_start_dt),
                np.datetime64(args.train_end_dt)
            ),
            second_slice=slice(
                args.start_second, args.end_second, args.step_second
            ),
            **dataset_kwargs,
        ),
        val_dataset=HDF5CSDataset(
            date_slice=slice(
                np.datetime64(args.val_start_dt),
                np.datetime64(args.val_end_dt)
            ),
            second_slice=slice(
                args.start_second, args.end_second, args.step_second
            ),
            **dataset_kwargs,
        ),
        test_dataset=HDF5CSDataset(
            date_slice=slice(
                np.datetime64(args.test_start_dt),
                np.datetime64(args.test_end_dt)
            ),
            # 测试集中的步长必须为1
            second_slice=slice(
                args.start_second, args.end_second, 1
            ),
            **dataset_kwargs,
        ),
    )

    # 加载模型
    model_cls: Type[CSModel] = import_object(args.model.class_path)
    init_args: Dict[str, Any] = args.model.init_args

    sig = inspect.signature(model_cls)
    if "x_fields" in sig.parameters:
        init_args["x_fields"] = engine.train_dataset.x_fields
    
    if "y_fields" in sig.parameters:
        init_args["y_fields"] = engine.train_dataset.y_fields
    
    if "seq_len" in sig.parameters:
        init_args["seq_len"] = args.seq_len

    engine.set_model(model_cls(**init_args))

    # 训练, 测试并执行其他任务
    if args.train:
        engine.train()

    engine.load_model(ckpt=args.test_ckpt)
    if args.test:
        engine.test()

    for task in args.task:
        engine.run(task)

    engine.close()


if __name__ == "__main__":
    args: jsonargparse.Namespace = get_args()
    run_once(args)

    base_folder: str = args.engine.folder

    # 使用全量数据进行训练
    if args.train_all:
        args.train = True
        args.engine.create_folder = True
        args.train_end_dt = args.val_end_dt
        args.metric_name = None

        min_ckpt: str = get_ckpt(args.engine.folder, "min")
        min_step: int = int(min_ckpt.split('_')[1].split('.')[0])
        max_ckpt: str = get_ckpt(args.engine.folder, "max")
        max_step: int = int(max_ckpt.split('_')[1].split('.')[0])
        args.max_n_val = validate_float_to_int(max_step / min_step)   

        for seed in args.train_all_seed:
            args.engine.folder = os.path.join(
                base_folder, f"train_all_sd{seed}"
            )
            args.engine.params["seed"] = seed
            run_once(args)
