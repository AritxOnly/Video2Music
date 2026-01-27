import logging
from logging.handlers import RotatingFileHandler
import datetime
import os
import sys


class StreamToLogger:
    """把 print() / stderr 输出重定向到 logging。"""
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        # 兼容 print 的分段写入
        message = message.rstrip("\n")
        if not message:
            return
        self.logger.log(self.level, message)

    def flush(self):
        pass


def setup_logging(log_dir: str = "logs", prefix: str = "video2music") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")

    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # 文件输出（滚动）
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # 避免重复添加 handler
    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)

    # 把 print / stderr 重定向到 logger（可选，但你希望“除了终端输出以外，还给我log文件”，这一步很关键）
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    logger.info(f"[Logging] Log file: {log_path}")
    return logger