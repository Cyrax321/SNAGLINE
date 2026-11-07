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

"""Bounded event buffer with eviction policies."""

from __future__ import annotations

from typing import Any


class BoundedEventBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items: list[Any] = []

    def push(self, item: Any) -> bool:
        evicted = False
        if len(self.items) >= self.capacity:
            self.items.pop(0)
            evicted = True
        self.items.append(item)
        return evicted

    def clear(self) -> None:
        """Atomically reset items without leaving stale references."""
        self.items.clear()
