# Copyright 2026 SNAGLINE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
