"""
系统IO相关的辅助工具
"""

import logging
import os
from typing import List


def get_logger(
    path: str,
    name: str = "main",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建一个logger同时输出到文件和控制台
    """

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(
        os.path.join(path, "main.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def reset_logger(logger: logging.Logger) -> None:
    """
    清除所有的logger及其handle
    """

    logger.propagate: bool = False
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def read_txt_lines(file: str) -> List[str]:
    """
    从一个txt文件路径中读取所有行

    每行对应一个item
    """
    
    with open(file, 'r') as f:
        contents: List[str] = f.readlines()
        contents = [content.strip() for content in contents]
    return contents


def write_txt_lines(file: str, contents: List[str]) -> None:
    """
    将一个列表写入到txt中

    每行对应一个item
    """

    with open(file, 'w') as f:
        for content in contents:
            f.write(content)
            f.write('\n')
