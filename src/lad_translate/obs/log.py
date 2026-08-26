"""
Structured logging.

One JSON object per line, with `severity` and all fields at the top level.
Cloud Run parses that shape directly into Cloud Logging, so fields stay
queryable. Both VOAG and MAGe format a human string and hand it to print or
console.log, which Cloud Logging stores as opaque text: readable in a terminal,
useless for building a latency dashboard from.

Never call print() in a production path. There is no fallback here that will
quietly make it work.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# Python level names to the strings Cloud Logging understands.
_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Renders a record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": _SEVERITY.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
        }
        # Anything passed via extra= lands at the top level, so it is queryable.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure(level: str | None = None) -> None:
    """Install the JSON handler on the root logger. Call once at start-up."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))


class ContextLogger(logging.LoggerAdapter):
    """
    Logger that merges bound context into every record.

    The stdlib LoggerAdapter REPLACES the per-call `extra` with the adapter's
    own, which silently drops the fields you actually wanted. This merges
    instead, with the per-call fields winning.
    """

    def process(self, msg: object, kwargs: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        merged = dict(self.extra or {})
        merged.update(kwargs.get("extra") or {})
        kwargs["extra"] = merged
        return msg, kwargs

    def bind(self, **fields: Any) -> "ContextLogger":
        """Return a logger carrying these fields on every record."""
        merged = dict(self.extra or {})
        merged.update(fields)
        return ContextLogger(self.logger, merged)


def get_logger(name: str, **context: Any) -> ContextLogger:
    """
    Logger for a module. Pass context with extra=, not by formatting it in.

        log = get_logger(__name__).bind(session_id=sid, tenant_id=tid)
        log.info("chunk committed", extra={"chunk_id": 4, "reason": "clause"})

    Bound context appears on every record, so a session can be filtered out of
    Cloud Logging without grepping message text.
    """
    return ContextLogger(logging.getLogger(name), context)
