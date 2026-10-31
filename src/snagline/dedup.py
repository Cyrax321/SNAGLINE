"""Fingerprint deduplication for event payloads.

Maintains an in-memory hash set of normalized payload digests to suppress
duplicate consecutive transmissions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class EventDeduplicator:
    """In-memory payload deduplication filter."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def is_duplicate(self, payload: Any) -> bool:
        """Check if payload digest has already been observed."""
        normalized = json.dumps(payload, sort_keys=True)
        h = hashlib.sha256(normalized.encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def clear(self) -> None:
        """Reset seen hashes to release fingerprint cache memory."""
        self._seen_hashes.clear()
