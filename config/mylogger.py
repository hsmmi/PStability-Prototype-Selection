import datetime as dt
import json
import logging
from typing import override
from tqdm import tqdm


class MyJsonFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str]):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_keys = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }
        if record.exc_info:
            always_keys["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            always_keys["stack"] = self.formatStack(record.stack_info)
        message = {
            key: (
                msg_val
                if (msg_val := always_keys.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_keys)

        # Extra
        for key, val in record.__dict__.items():
            if key not in message:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    @override
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class TqdmFilter(logging.Filter):
    def __init__(self, include=False):
        super().__init__()
        self.include = include

    @override
    def filter(self, record):
        use_tqdm = getattr(record, "use_tqdm", False)
        return use_tqdm if self.include else not use_tqdm
