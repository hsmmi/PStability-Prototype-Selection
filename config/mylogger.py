import datetime as dt
import json
import logging
from typing import override


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
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
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
