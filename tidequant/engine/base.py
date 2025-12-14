"""
负责执行任务的核心引擎
"""

import logging
import os
import shutil
from abc import ABC, ABCMeta
from typing import Any, Dict, Tuple, Type

from torch._decomp.decompositions import type_casts

from ..utils.io import get_logger, reset_logger


registry: Dict[str, Type["Engine"]] = {}


class EngineMeta(ABCMeta):
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
        registry[name] = cls
        return cls


class Engine(ABC, metaclass=EngineMeta):
    """
    引擎基类
    
    引擎运行的结果会被写入一个固定的文件夹
    """

    # 业务/算法参数
    params: Dict[str, Any] = {}

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        params: Dict[str, Any] | None = None,
    ) -> None:
        self.folder: str = folder

        _params: Dict[str, Any] = self.params.copy()
        if params is not None:
            _params.update(params)
        self.params = _params

        if create_folder:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

        self.logger: logging.Logger = get_logger(folder)
        self.logger.info(f"params are {self.params}")

    def close(self, ) -> None:
        """
        释放资源并关闭引擎
        """
        reset_logger(self.logger)
