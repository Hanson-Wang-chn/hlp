# -*- coding: utf-8 -*-
"""
日志模块
提供统一的日志记录功能，支持结构化日志输出
"""

import time
import uuid
import json
import logging
import numpy as np
from typing import Dict, Any, Optional


class NumpyJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，支持numpy类型序列化
    """

    def default(self, obj):
        """
        处理numpy类型转换为Python原生类型

        Args:
            obj: 要序列化的对象

        Returns:
            可JSON序列化的Python对象
        """
        # numpy整数类型
        if isinstance(obj, np.integer):
            return int(obj)
        # numpy浮点类型
        elif isinstance(obj, np.floating):
            return float(obj)
        # numpy数组
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy布尔类型
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def now_ms() -> int:
    """
    获取当前毫秒级时间戳

    Returns:
        毫秒级epoch时间戳
    """
    return int(time.time() * 1000)


def new_run_id() -> str:
    """
    生成新的运行ID

    Returns:
        UUID4字符串
    """
    return str(uuid.uuid4())


def make_logger(cfg: Dict[str, Any]) -> logging.Logger:
    """
    创建日志器

    Args:
        cfg: 配置字典，需包含 cfg["log"] 配置

    Returns:
        配置好的Logger对象
    """
    logger = logging.getLogger("HLP")

    # 获取日志级别
    level_str = cfg.get("log", {}).get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # 是否打印到控制台
    print_to_console = cfg.get("log", {}).get("print_to_console", True)

    if print_to_console and not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(level)
        fmt = logging.Formatter("%(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)

    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    run_id: str,
    step_id: Optional[int],
    module: str,
    event: str,
    status: str = "ok",
    elapsed_ms: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    structured: bool = True
) -> None:
    """
    记录结构化日志事件

    Args:
        logger: Logger对象
        run_id: 运行ID
        step_id: 步骤ID
        module: 模块名称 ("HLP", "M1", "M2", "M3", "M4", "M5")
        event: 事件类型 ("start", "end", "cache_hit", "cache_miss", "warn", "error")
        status: 状态 ("ok", "empty", "fail")
        elapsed_ms: 耗时（毫秒）
        details: 扩展信息（必须是JSON可序列化）
        structured: 是否输出结构化格式
    """
    payload = {
        "ts_ms": now_ms(),
        "run_id": run_id,
        "step_id": step_id,
        "module": module,
        "event": event,
        "elapsed_ms": elapsed_ms,
        "status": status,
        "details": details or {}
    }

    if structured:
        logger.info(json.dumps(payload, ensure_ascii=False, cls=NumpyJSONEncoder))
    else:
        # 非结构化格式，更易读
        msg_parts = [f"[{module}]", f"event={event}", f"status={status}"]
        if elapsed_ms is not None:
            msg_parts.append(f"elapsed={elapsed_ms:.1f}ms")
        if details:
            msg_parts.append(f"details={details}")
        logger.info(" ".join(msg_parts))


class HLPLogger:
    """
    HLP日志封装类，方便模块调用
    """

    def __init__(self, logger: logging.Logger, run_id: str, structured: bool = True):
        """
        初始化HLP日志器

        Args:
            logger: Python Logger对象
            run_id: 运行ID
            structured: 是否使用结构化日志
        """
        self.logger = logger
        self.run_id = run_id
        self.structured = structured

    def log(
        self,
        module: str,
        event: str,
        step_id: Optional[int] = None,
        status: str = "ok",
        elapsed_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录日志事件

        Args:
            module: 模块名称
            event: 事件类型
            step_id: 步骤ID
            status: 状态
            elapsed_ms: 耗时
            details: 扩展信息
        """
        log_event(
            self.logger,
            self.run_id,
            step_id,
            module,
            event,
            status,
            elapsed_ms,
            details,
            self.structured
        )

    def start(self, module: str, step_id: Optional[int] = None) -> float:
        """
        记录模块开始，返回开始时间

        Args:
            module: 模块名称
            step_id: 步骤ID

        Returns:
            开始时间戳（秒）
        """
        self.log(module, "start", step_id=step_id)
        return time.time()

    def end(
        self,
        module: str,
        start_time: float,
        step_id: Optional[int] = None,
        status: str = "ok",
        details: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        记录模块结束

        Args:
            module: 模块名称
            start_time: 开始时间戳（秒）
            step_id: 步骤ID
            status: 状态
            details: 扩展信息

        Returns:
            耗时（毫秒）
        """
        elapsed_ms = (time.time() - start_time) * 1000
        self.log(module, "end", step_id=step_id, status=status,
                 elapsed_ms=elapsed_ms, details=details)
        return elapsed_ms

    def cache_hit(self, module: str, step_id: Optional[int] = None) -> None:
        """记录缓存命中"""
        self.log(module, "cache_hit", step_id=step_id)

    def cache_miss(self, module: str, step_id: Optional[int] = None) -> None:
        """记录缓存未命中"""
        self.log(module, "cache_miss", step_id=step_id)

    def warn(self, module: str, message: str, step_id: Optional[int] = None) -> None:
        """记录警告"""
        self.log(module, "warn", step_id=step_id, status="warn",
                 details={"message": message})

    def error(self, module: str, message: str, step_id: Optional[int] = None) -> None:
        """记录错误"""
        self.log(module, "error", step_id=step_id, status="fail",
                 details={"message": message})
