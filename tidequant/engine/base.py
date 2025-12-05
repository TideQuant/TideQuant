"""
负责执行任务的核心引擎
"""

import argparse
import inspect
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

from ..utils import get_logger, reset_logger


registry: Dict[str, ] = {}


class EngineMeta(type):
    """
    Engine的子类都会被注册到表中
    """

    def __new__(
        mcls: type,
        name: str,
        bases: Tuple[type, ...],
        attrs: Dict[str, Any],
    ) -> "EngineMeta":
        cls = super().__new__(mcls, name, bases, attrs)

        # 不注册抽象类
        if not inspect.isabstract(cls):
            registry[name] = cls
        return cls


class Engine(ABC, metaclass=EngineMeta):
    """
    引擎基类
    
    引擎运行的结果会被写入一个固定的文件夹

    支持添加args参数和从args中实例化类
    尽量避免在params中指定默认值为None, 因为它们不会被加入到argparse中
    """

    # 业务/算法参数
    params: Dict[str, Any] = {}

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        params: Dict[str, Any] = {},
    ) -> None:
        self.folder: str = folder

        _params = self.params.copy()
        _params.update(params)
        self.params = _params

        if create_folder:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

        self.logger: logging.Logger = get_logger(folder)
        self.logger.info(f"params are {self.params}")

    @classmethod
    def add_args(
        cls: Type["Engine"],
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """
        将params中的default不为None的参数加入到argparse中
        """
        for name, default in cls.params.items():

            if isinstance(default, bool):
                parser.add_argument(
                    arg_name,
                    type=str2bool,
                    default=default,
                    help=f"(bool) default: {default}",
                )
            elif isinstance(default, int):
                parser.add_argument(
                    arg_name,
                    type=int,
                    default=default,
                    help=f"(int) default: {default}",
                )
            elif isinstance(default, float):
                parser.add_argument(
                    arg_name,
                    type=float,
                    default=default,
                    help=f"(float) default: {default}",
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=str,
                    default=default,
                    help=f"(str) default: {default}",
                )
        return parser

    @classmethod
    def from_args(
        cls: Type["Engine"],
        args: argparse.Namespace,
    ) -> "Engine":
        """
        根据args创建实例
        """
        args_dict: Dict[str, Any] = vars(args).copy()
        args_dict = {k: v for k, v in args_dict.items() if k in cls.params or cls.}

        return cls(
            folder=args_dict.pop("folder"),
            create_folder=args_dict.pop("create_folder"),
            params=args_dict,
        )

    @abstractmethod
    def run(self, ) -> None:
        """
        程序主入口, 执行任务
        """
        pass

    def close(self, ) -> None:
        """
        释放资源并关闭引擎
        """
        reset_logger(self.logger)
