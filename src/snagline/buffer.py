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
