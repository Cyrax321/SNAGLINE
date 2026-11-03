"""Tamper-evident security authorization audit log schema.

Maintains an append-only, SHA-256 hash-chained ledger of security authorization
decisions, rule findings, and request metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Decision = Literal["allow", "ask", "deny"]
GENESIS_HASH: str = "0" * 64


class AuditLogError(Exception):
    """Raised when an audit log write or verification fails in fail-closed mode."""

    pass


@dataclass
class AuditRecord:
    """A single immutable entry in the tamper-evident authorization audit log."""

    index: int
    timestamp: str
    decision: Decision
    tool_name: str
    policy_id: str
    rule_findings: list[dict[str, Any]] = field(default_factory=list)
    request_meta: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def canonical_payload(self) -> str:
        """Produce deterministic canonical JSON representation of record fields."""
        data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "tool_name": self.tool_name,
            "policy_id": self.policy_id,
            "rule_findings": self.rule_findings,
            "request_meta": self.request_meta,
            "previous_hash": self.previous_hash,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the canonical payload."""
        payload = self.canonical_payload().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        """Construct record from dictionary."""
        return cls(
            index=int(data["index"]),
            timestamp=str(data["timestamp"]),
            decision=data["decision"],
            tool_name=str(data["tool_name"]),
            policy_id=str(data["policy_id"]),
            rule_findings=list(data.get("rule_findings", [])),
            request_meta=dict(data.get("request_meta", {})),
            previous_hash=str(data.get("previous_hash", GENESIS_HASH)),
            entry_hash=str(data.get("entry_hash", "")),
        )


class SecurityAuditLog:
    """Thread-safe append-only tamper-evident audit ledger."""

    def __init__(
        self,
        log_path: str | Path | None = None,
        fail_closed: bool = False,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.fail_closed = fail_closed
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._last_hash = GENESIS_HASH

        if self.log_path and self.log_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing records from log_path and verify their hash chain.

        Malformed lines are skipped unless ``fail_closed=True``. The parsed
        records must also form an untampered chain: a history edit changes the
        hashed content while leaving the stored hashes in place, so JSON
        decoding alone cannot tell a pristine log from a rewritten one. A
        broken chain raises :class:`AuditLogError` under ``fail_closed=True``
        (tamper evidence is the ledger's purpose) and is logged but still
        loaded in the fail-open mode, exactly as malformed lines are.
        """
        if not self.log_path:
            return
        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        record = AuditRecord.from_dict(raw)
                        self._records.append(record)
                        self._last_hash = record.entry_hash
                    except Exception as err:
                        if self.fail_closed:
                            raise AuditLogError(
                                f"Malformed audit log entry at line {line_num}: {err}"
                            ) from err
        except OSError as err:
            if self.fail_closed:
                raise AuditLogError(
                    f"Failed to read audit log {self.log_path}: {err}"
                ) from err

        is_valid, fault = self.verify()
        if is_valid:
            return
        if self.fail_closed:
            raise AuditLogError(
                f"Loaded audit log {self.log_path} fails chain verification: {fault}"
            )
        logger.warning(
            "Snagline audit log %s fails chain verification (%s); loading it "
            "anyway because fail_open is enabled. Call verify() on the log "
            "before trusting its history.",
            self.log_path,
            fault,
        )

    def append(
        self,
        decision: Decision,
        tool_name: str,
        policy_id: str = "default",
        rule_findings: list[dict[str, Any]] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Append an authorization decision to the audit log."""
        with self._lock:
            clean_meta = self._sanitize_meta(request_meta or {})
            clean_findings = self._sanitize_findings(rule_findings or [])

            timestamp = datetime.now(timezone.utc).isoformat()
            index = len(self._records)
            prev_hash = self._last_hash

            record = AuditRecord(
                index=index,
                timestamp=timestamp,
                decision=decision,
                tool_name=tool_name,
                policy_id=policy_id,
                rule_findings=clean_findings,
                request_meta=clean_meta,
                previous_hash=prev_hash,
            )
            record.entry_hash = record.compute_hash()

            if self.log_path:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record.to_dict()) + "\n")
                except OSError as err:
                    if self.fail_closed:
                        raise AuditLogError(
                            f"Failed to write to audit log {self.log_path}: {err}"
                        ) from err

            self._records.append(record)
            self._last_hash = record.entry_hash
            return record

    @staticmethod
    def _sanitize_meta(meta: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive credentials, authorization tokens, and secrets from meta."""
        sanitized: dict[str, Any] = {}
        for k, v in meta.items():
            k_lower = str(k).lower()
            if any(
                s in k_lower
                for s in ("token", "secret", "password", "key", "auth", "credential")
            ):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = SecurityAuditLog._sanitize_meta(v)
            else:
                sanitized[k] = v
        return sanitized

    @staticmethod
    def _sanitize_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize findings to ensure raw secrets or credentials are not preserved."""
        cleaned = []
        for f in findings:
            item = dict(f)
            if "raw" in item:
                del item["raw"]
            cleaned.append(item)
        return cleaned

    def verify(self) -> tuple[bool, str | None]:
        """Verify the integrity of the hash chain across all records."""
        with self._lock:
            expected_prev = GENESIS_HASH
            for i, record in enumerate(self._records):
                if record.index != i:
                    return (
                        False,
                        f"Broken sequence index at entry {i}: expected {i}, got {record.index}",
                    )
                if record.previous_hash != expected_prev:
                    return False, (
                        f"Hash chain broken at entry {i}: expected previous_hash {expected_prev}, "
                        f"got {record.previous_hash}"
                    )
                computed = record.compute_hash()
                if record.entry_hash != computed:
                    return False, (
                        f"Tampered content at entry {i}: recorded hash {record.entry_hash} "
                        f"does not match computed hash {computed}"
                    )
                expected_prev = record.entry_hash
            return True, None

    @property
    def records(self) -> list[AuditRecord]:
        """Return copy of recorded entries."""
        with self._lock:
            return list(self._records)


def verify_audit_log_file(file_path: str | Path) -> tuple[bool, str | None]:
    """Standalone utility to verify hash-chain integrity of an audit log file on disk."""
    path = Path(file_path)
    if not path.exists():
        return False, f"Audit log file not found: {path}"

    expected_prev = GENESIS_HASH
    try:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = AuditRecord.from_dict(data)
                except Exception as err:
                    return False, f"Malformed JSON at line {line_num + 1}: {err}"

                if record.index != line_num:
                    return (
                        False,
                        f"Broken index at line {line_num + 1}: expected {line_num}, got {record.index}",
                    )
                if record.previous_hash != expected_prev:
                    return False, (
                        f"Broken chain link at line {line_num + 1}: expected previous {expected_prev}, "
                        f"got {record.previous_hash}"
                    )
                computed = record.compute_hash()
                if record.entry_hash != computed:
                    return False, (
                        f"Hash mismatch at line {line_num + 1}: computed {computed}, "
                        f"recorded {record.entry_hash}"
                    )
                expected_prev = record.entry_hash
        return True, None
    except OSError as err:
        return False, f"Failed reading audit log: {err}"
