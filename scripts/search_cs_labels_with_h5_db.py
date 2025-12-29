"""
用于搜索标签的脚本
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import json
import time
from typing import Any, Dict, List

import jsonargparse
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from train_cs_model_with_h5_db import get_parser, run_once


def search(config: Dict[str, Any]) -> None:
    _args: jsonargparse.ArgumentParser = copy.copy(args)
    _args.train = True
    _args.train_all = False
    _args.test = False
    _args.engine.folder = os.path.join(_args.folder, str(time.time()))
    _args.engine.create_folder = True
    # 将早停的控制权限交给ray
    _args.patience = 100
    _args.rt_report = True

    # 根据权重计算y_fields
    weights = np.array([config[f"w{i}"] for i in range(len(_args.all_yields))])
    chosen: List[int] = np.argsort(weights)[-config["topk"]: ][:: -1].tolist()
    _args.y_fields = [
        _args.y_field,
    ] + np.array(_args.all_yields)[chosen].tolist()

    # TODO: 加锁写入共享结果文件会更安全
    best_metric: float = run_once(_args)
    line: str = json.dumps(
        {"metric": best_metric, "y_fields": _args.y_fields}, ensure_ascii=False
    ) + "\n"
    fd = os.open(
        os.path.join(_args.folder, "results.json"),
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    os.write(fd, line.encode("utf-8"))
    os.close(fd)


args: jsonargparse.ArgumentParser = None


if __name__ == "__main__":
    parser: jsonargparse.ArgumentParser = get_parser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--y_corr_csv", type=str, required=True)
    parser.add_argument("--y_field", type=str, required=True)
    parser.add_argument("--threhold", type=float, default=0.95)
    # 忽略这两个必填属性, 下面会赋值
    args = parser.parse_args(
        sys.argv[1:]
        + ["--y_fields", "__dummy__", "--engine.folder", "__dummy__"]
    )

    # 获取候选y
    y_corr: pd.DataFrame = pd.read_csv(args.y_corr_csv, index_col=0)
    cand = y_corr.index[
        (y_corr[args.y_field].abs() < args.threhold)
        & (y_corr.index != args.y_field)
    ]
    args.all_yields = []
    for field in cand:
        if all(y_corr.loc[field, args.all_yields] < args.threhold):
            args.all_yields.append(field)

    # 构造space
    space: Dict[str, float] = {
        f"w{i}": tune.uniform(0.0, 1.0) for i in range(len(args.all_yields))
    }
    space["topk"] = tune.randint(3, 5)

    tuner = tune.Tuner(
    tune.with_resources(search, {"cpu": 8, "gpu": 1}),
        param_space=space,
        tune_config=tune.TuneConfig(
            metric=args.metric_name, mode="max", num_samples=1000,
            search_alg=OptunaSearch(
                sampler=TPESampler(
                    n_startup_trials=30,
                    constant_liar=True,
                    seed=42,
                ),
            ),
            scheduler=ASHAScheduler(
                max_t=20,
                grace_period=10,
                reduction_factor=2,
            ),
        ),
        run_config=air.RunConfig(
            name="search",
            stop={"training_iteration": 20},
            verbose=0,
        ),
    )

    res = tuner.fit()
