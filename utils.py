"""工具函数：环境变量读取 + 统一日志配置"""

import os
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """读取环境变量"""
    return os.getenv(key, default)


# ==================== 统一日志配置 ====================

class ColoredFormatter(logging.Formatter):
    """控制台彩色日志格式化"""

    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = "medagent",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    配置统一日志

    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径，None 则只输出到控制台
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 统一格式
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 控制台处理器（彩色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(fmt, datefmt=datefmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器（轮转）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 阻止日志向上传播到 root logger
    logger.propagate = False

    return logger


# 默认日志器
_default_logger = None

def get_logger(name: str = "medagent") -> logging.Logger:
    """获取默认配置好的日志器"""
    global _default_logger
    if _default_logger is None:
        log_level = get_env("LOG_LEVEL", "INFO")
        log_file = get_env("LOG_FILE", "./logs/medagent.log")
        _default_logger = setup_logger(name, log_level, log_file)
    return logging.getLogger(name)