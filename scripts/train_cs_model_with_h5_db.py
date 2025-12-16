"""
训练截面模型的脚本

支持预训练

支持训练确定参数后再用训练集+验证集训练一遍
"""

import inspect
import jsonargparse
import sys
from jsonargparse.util import import_object
from typing import Any, Dict, List, Type

import numpy as np

from tidequant.dl import (
    AccelerateEngine,
    Callback,
    CSModel,
    EarlyStopSaver,
    HDF5CSDataset,
    WarmUpSchedule,
    get_newest_ckpt,
    get_oldest_ckpt,
)
from tidequant.utils import validate_float_to_int


def get_args() -> jsonargparse.Namespace:
    """
    从命令行获取参数
    """

    parser = jsonargparse.ArgumentParser()
    
    # 配置运行参数
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_all_data", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument("--task", nargs='+')
    
    # 配置AccelerateEngine
    parser.add_class_arguments(
        AccelerateEngine,
        "engine",
        skip={"create_folder", "callbacks"},
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
    parser.add_argument("--y_fields", nargs='+')
    # 这里默认是Bar1m的配置
    parser.add_argument("--train_start_dt", type=str, default="2020-01-01")
    parser.add_argument("--train_end_dt", type=str, default="2024-10-31")
    parser.add_argument("--val_start_dt", type=str, default="2024-11-01")
    parser.add_argument("--val_end_dt", type=str, default="2024-12-31")
    parser.add_argument("--test_start_dt", type=str, default="2024-11-01")
    parser.add_argument("--test_end_dt", type=str, default="2024-12-31")
    parser.add_argument("--second_slice", type=lambda s: eval(s))
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
        create_folder=args.train,
        callbacks=callbacks,
        params=args.engine.params,
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
            second_slice=args.second_slice,
            **dataset_kwargs,
        ),
        val_dataset=HDF5CSDataset(
            date_slice=slice(
                np.datetime64(args.val_start_dt),
                np.datetime64(args.val_end_dt)
            ),
            second_slice=args.second_slice,
            **dataset_kwargs,
        ),
        test_dataset=HDF5CSDataset(
            date_slice=slice(
                np.datetime64(args.test_start_dt),
                np.datetime64(args.test_end_dt)
            ),
            # 测试集中的步长必须为1
            second_slice=slice(
                args.second_slice.start, args.second_slice.stop, 1
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


if __name__ == "__main__":
    args: jsonargparse.Namespace = get_args()
    run_once(args)

    # 使用全量数据进行训练
    if args.train_all:
        # 文件夹名后面加_all
        args.folder = (
            args.folder if args.folder[-1] != '/' else args.folder[: -1]
        ) + "_all"
        args.train_end_dt = args.val_end_dt
        args.metric_name = None

        oldest_ckpt: str = get_oldest_ckpt(args.engine.folder)
        oldest_step: int = oldest_ckpt.split('_')[1].split('.')[0]
        newest_ckpt: str = get_newest_ckpt(args.engine.folder)
        newest_step: int = newest_ckpt.split('_')[1].split('.')[0]
        args.max_n_val = validate_float_to_int(newest_step / oldest_step)     

        run_once(args)
