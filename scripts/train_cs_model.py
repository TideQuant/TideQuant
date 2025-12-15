# coding: utf-8
"""\
??AccelerateEngine??????, ????jsonargparse??.

model/callback??????:
1) name: ????, ??MLP
2) class_path: ??????
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List

import numpy as np

try:
    import jsonargparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "??jsonargparse, ????: pip install jsonargparse"
    ) from exc

from tidequant.dl.callback import Callback
from tidequant.dl.dataset import HDF5CSDataset
from tidequant.dl.engine import AccelerateEngine


_MODEL_NAME_TO_PATH: Dict[str, str] = {
    "MLP": "tidequant.dl.model.cs_models.mlp.MLP",
}

_CALLBACK_NAME_TO_PATH: Dict[str, str] = {
    "WarmUpSchedule": "tidequant.dl.callback.WarmUpSchedule",
    "EarlyStopSaver": "tidequant.dl.callback.EarlyStopSaver",
    "RayTuneReport": "tidequant.dl.callback.RayTuneReport",
}


def _parse_slice(spec: Any, to_dt: bool) -> slice:
    if spec is None:
        return slice(None)

    if isinstance(spec, slice):
        return spec

    if isinstance(spec, dict):
        start, stop, step = spec.get("start"), spec.get("stop"), spec.get("step")
    elif isinstance(spec, (list, tuple)) and len(spec) == 3:
        start, stop, step = spec
    else:
        raise ValueError("slice?????dict?3??list")

    if to_dt:
        start = None if start is None else np.datetime64(start)
        stop = None if stop is None else np.datetime64(stop)

    return slice(start, stop, step)


def _build_dataset(cfg: Dict[str, Any]) -> HDF5CSDataset:
    cfg = dict(cfg)

    date_spec = cfg.pop("date_slice", None)
    if date_spec is None:
        start_dt = cfg.pop("start_dt", None)
        end_dt = cfg.pop("end_dt", None)
        date_step = cfg.pop("date_step", None)
        if start_dt is not None or end_dt is not None or date_step is not None:
            date_spec = [start_dt, end_dt, date_step]

    second_spec = cfg.pop("second_slice", None)
    if second_spec is None:
        start_second = cfg.pop("start_second", None)
        end_second = cfg.pop("end_second", None)
        second_step = cfg.pop("second_step", None)
        if (
            start_second is not None
            or end_second is not None
            or second_step is not None
        ):
            second_spec = [start_second, end_second, second_step]

    cfg["date_slice"] = _parse_slice(date_spec, to_dt=True)
    cfg["second_slice"] = _parse_slice(second_spec, to_dt=False)
    return HDF5CSDataset(**cfg)


def _instantiate(
    spec: Dict[str, Any],
    name_to_path: Dict[str, str],
    extra_kwargs: Dict[str, Any] | None = None,
) -> Any:
    if "class_path" in spec:
        class_path = spec["class_path"]
    else:
        name = spec.get("name")
        if name is None:
            raise ValueError("??????name?class_path")
        class_path = name_to_path.get(name, name)

    init_args = spec.get("init_args") or {}
    if not isinstance(init_args, dict):
        raise TypeError("init_args?????")

    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)

    kwargs = dict(init_args)
    if extra_kwargs:
        for k, v in extra_kwargs.items():
            kwargs.setdefault(k, v)

        try:
            sig = inspect.signature(cls.__init__)
            if not any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            ):
                allowed = set(sig.parameters)
                allowed.discard("self")
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except (TypeError, ValueError):
            pass

    return cls(**kwargs)


def main(args: jsonargparse.Namespace) -> None:
    params: Dict[str, Any] = dict(AccelerateEngine.params)
    params.update(args.params or {})

    train_dataset = _build_dataset(args.train_dataset)
    val_dataset = _build_dataset(args.val_dataset)

    callbacks: List[Callback] = []
    for spec in args.callbacks or []:
        cb = _instantiate(spec, _CALLBACK_NAME_TO_PATH)
        if not hasattr(cb, "params"):
            setattr(cb, "params", params)
        callbacks.append(cb)

    model = _instantiate(
        args.model,
        _MODEL_NAME_TO_PATH,
        extra_kwargs={
            "x_fields": train_dataset.x_fields,
            "y_fields": train_dataset.y_fields,
        },
    )

    engine = AccelerateEngine(
        folder=args.folder,
        create_folder=args.create_folder,
        callbacks=callbacks,
        params=params,
    )
    engine.set_dataset(train_dataset=train_dataset, val_dataset=val_dataset)
    engine.set_model(model)

    if args.ckpt:
        engine.load_model(args.ckpt)

    engine.train()
    engine.close()


def _build_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action=jsonargparse.ActionConfigFile,
        help="??????, ??yaml?json",
    )
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--create_folder", type=bool, default=True)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--params", type=Dict[str, Any], default=None)
    parser.add_argument("--train_dataset", type=Dict[str, Any], required=True)
    parser.add_argument("--val_dataset", type=Dict[str, Any], required=True)
    parser.add_argument("--model", type=Dict[str, Any], required=True)
    parser.add_argument("--callbacks", type=List[Dict[str, Any]], default=None)
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
