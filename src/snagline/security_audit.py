"""Tamper-evident security authorization audit log schema.

Maintains an append-only, SHA-256 hash-chained ledger of security authorization
decisions, rule findings, and request metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

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
    rule_findings: List[Dict[str, Any]] = field(default_factory=list)
    request_meta: Dict[str, Any] = field(default_factory=dict)
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditRecord:
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
