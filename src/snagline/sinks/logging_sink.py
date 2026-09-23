"""Logging sink -- route risks through the stdlib ``logging`` module (issue #99).

Log aggregators want machine-readable JSON, one object per line. This sink
emits exactly one ``logging`` record per ``FailureRisk`` on the ``"snagline"``
logger; its formatter renders a single compact JSON object with EXACTLY these
keys: ``ts``, ``episode_id``, ``step_id``, ``trigger``, ``severity``,
``score``, ``detail``.

Structure only, never content: every value comes from the ``FailureRisk``
itself, which by design carries no prompt or response text (project.md §11),
so a log pipeline cannot become a data-exfiltration path.

Fail-open: if serialization raises, the formatter falls back to a plain
structural line and the sink swallows everything else, mirroring every other
sink's fire-and-forget contract (issue #19).
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress

from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")

# Compact separators keep one risk to a single line even when detail is long.
_COMPACT_SEPARATORS = (",", ":")


class JsonRiskFormatter(logging.Formatter):
    """Render a ``FailureRisk`` as one single-line JSON object.

    Works two ways:

    * ``render(risk)`` -- serialize directly, used by :class:`LoggingSink`.
    * ``format(record)`` -- standard ``logging.Formatter`` entry point for
      handlers that want to re-render records carrying the risk in
      ``record.snagline_risk``; other records fall back to default formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        risk = getattr(record, "snagline_risk", None)
        if isinstance(risk, FailureRisk):
            return self.render(risk)
        return super().format(record)

    def render(self, risk: FailureRisk) -> str:
        """Serialize one risk to a compact JSON line (fail-open)."""
        try:
            return self._serialize(risk, ensure_ascii=False)
        except Exception:
            return self._fallback(risk)

    def render_ascii(self, risk: FailureRisk) -> str:
        """ASCII-escaped variant of :meth:`render`, for streams that cannot hold
        non-ASCII text (issue #431).

        ``render`` keeps the common UTF-8 case readable by emitting non-ASCII
        raw, but a handler whose stream targets a narrower codepage fails
        inside ``StreamHandler.emit`` -- and ``logging``'s own ``handleError``
        absorbs that failure, printing a traceback and *dropping the record*.
        The sink never sees an exception to catch, so the alert vanishes and
        the per-alert traceback reads as a fault of the instrumentation. Every
        codepoint is escaped here instead, which any byte stream can carry;
        ``json.loads`` yields the identical string, so no pipeline loses data.
        """
        try:
            return self._serialize(risk, ensure_ascii=True)
        except Exception:
            return self._fallback(risk)

    def _serialize(self, risk: FailureRisk, *, ensure_ascii: bool) -> str:
        payload: dict[str, object] = {
            "ts": risk.timestamp,
            "episode_id": risk.episode_id,
            "step_id": risk.step_id,
            "trigger": risk.trigger,
            "severity": risk.severity,
            "score": risk.score,
            "detail": risk.detail,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=_COMPACT_SEPARATORS,
            ensure_ascii=ensure_ascii,
        )

    def _fallback(self, risk: FailureRisk) -> str:
        # Fail-open: never propagate a serialization failure into the host
        # agent. The fallback is plain text and ids only; ``detail`` is
        # deliberately excluded since it is the field most likely to have
        # broken encoding.
        return (
            "snagline risk (json encoding failed) "
            f"episode_id={risk.episode_id!r} step_id={risk.step_id!r} "
            f"trigger={risk.trigger!r} severity={risk.severity!r} "
            f"score={risk.score!r} ts={risk.timestamp!r}"
        )


def _stream_encodings(logger: logging.Logger) -> list[str]:
    """Encodings declared by the stream handlers on ``logger`` and its ancestors.

    ``logging`` propagates to ancestor loggers by default, so the root logger's
    stderr handler counts too -- on a host whose locale is a narrow ANSI codepage
    that is the handler that drops the record (issue #431).
    """
    found: list[str] = []
    node: logging.Logger | None = logger
    while node is not None:
        for handler in node.handlers:
            enc = getattr(getattr(handler, "stream", None), "encoding", None)
            if isinstance(enc, str):
                found.append(enc)
        node = node.parent
    return found


class LoggingSink:
    """Emit each ``FailureRisk`` as one record on the ``"snagline"`` logger.

    The formatted message is the JSON line itself, so any handler (and any
    downstream log shipper) receives one parseable object per line without
    extra configuration. Default level WARNING matches the console sink's
    logging mode.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.WARNING,
        formatter: JsonRiskFormatter | None = None,
    ) -> None:
        self._logger = logger if logger is not None else logging.getLogger("snagline")
        self._level = level
        self._formatter = formatter if formatter is not None else JsonRiskFormatter()

    def emit(self, risk: FailureRisk) -> None:
        try:
            line = self._formatter.render(risk)
        except Exception:
            # Fail-open belt and braces: even a broken custom formatter must
            # not crash ingest(); drop the alert instead (issue #19 pattern).
            logger.warning(
                "snagline LoggingSink: rendering failed; dropping alert "
                "(fire-and-forget)"
            )
            return
        # Fire-and-forget like every other AlertSink: a misconfigured logging
        # pipeline must never raise out of emit() (issue #19).
        # A stream that cannot encode the raw line would drop the record inside
        # logging's own handleError -- silently, since no exception escapes
        # Logger.log to land in the suppress below -- so fall back to the
        # ASCII-escaped variant first, when any attached stream needs it
        # (#431).
        for enc in _stream_encodings(self._logger):
            try:
                line.encode(enc)
            except (UnicodeEncodeError, LookupError):
                render_ascii = getattr(self._formatter, "render_ascii", None)
                if callable(render_ascii):
                    line = render_ascii(risk)
                break
        with suppress(Exception):  # pragma: no cover - host logger failure
            self._logger.log(self._level, "%s", line, extra={"snagline_risk": risk})
