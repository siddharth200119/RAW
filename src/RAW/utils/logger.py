import logging
import sys
import json
from datetime import datetime
from contextvars import ContextVar
from typing import Optional

_tracker_id: ContextVar[Optional[str]] = ContextVar("tracker_id", default=None)


class Logger:
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",
    }

    def __init__(self, service_name: str, level: int = logging.INFO):
        self.service_name = service_name
        self.is_tty = sys.stdout.isatty()

        self._logger = logging.getLogger(service_name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._formatter())

        self._logger.handlers.clear()
        self._logger.addHandler(handler)

    def _formatter(self):
        if self.is_tty:
            return self._colored_formatter()
        return self._json_formatter()

    def _colored_formatter(self):
        class ColoredFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                color = Logger.COLORS.get(record.levelname, "")
                reset = Logger.COLORS["RESET"]

                tracker_id = _tracker_id.get()
                tracker_part = f" | tracker_id={tracker_id}" if tracker_id else ""

                return (
                    f"{color}"
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{record.levelname:<8} | "
                    f"{record.name} | "
                    f"{record.getMessage()}"
                    f"{tracker_part}"
                    f"{reset}"
                )

        return ColoredFormatter()

    def _json_formatter(self):
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(timespec="milliseconds")
                        + "Z",
                        "level": record.levelname,
                        "service_name": record.name,
                        "message": record.getMessage(),
                        "tracker_id": _tracker_id.get(),
                    }
                )

        return JsonFormatter()

    # ---- static helpers ----
    @staticmethod
    def set_tracker_id(tracker_id: Optional[str]):
        _tracker_id.set(tracker_id)

    # ---- logging methods ----
    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)
