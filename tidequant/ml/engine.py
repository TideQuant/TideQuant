"""
机器学习实验基类
"""

import logging
import os
import shutil
from typing import Any, Dict

from abc import ABC, abstractmethod

from ..utils import get_logger, reset_logger, set_seed


class AccelerateEngine(ABC):
    """
    基于Accelerate的引擎基类
    """

    params: Dict[str, Any] = {}

    def __init__(self, ) -> None:
        pass
