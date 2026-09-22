"""A non-positive ``max_body_bytes`` rejects every POST (#394).

``do_POST`` compares ``length > snagline_max_body`` with a strict ``>``, so a
cap of 0 (or any negative) admits no body at all: the sidecar starts cleanly,
answers ``GET /health`` 200, and drops 100% of inbound telemetry with 413. The
neighbouring ``max_risks`` knob is clamped with ``max(1, ...)``; this one was
not, and there is no ``None`` escape hatch on the flag. Both entry points now
refuse the value at construction.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from snagline.monitor import Monitor
from snagline.server.http_server import make_handler, make_server

_EVENT = json.dumps(
    {
        "episode_id": "ep",
        "step_id": "s0",
        "action_type": "tool_call",
        "action_signature": "tool_call:search:s0",
        "timestamp": 1.0,
    }
).encode("utf-8")


@pytest.mark.parametrize("bad", [0, -1, -1_000_000])
def test_make_handler_rejects_a_non_positive_cap(bad: int) -> None:
    # The library entry point must refuse too: a caller who never goes near
    # the CLI would otherwise get a green-but-deaf server.
    with pytest.raises(ValueError, match="max_body_bytes"):
        make_handler(Monitor([], []), max_body_bytes=bad)


def test_make_server_rejects_a_non_positive_cap() -> None:
    with pytest.raises(ValueError, match="413"):
        make_server(Monitor([], []), host="127.0.0.1", port=0, max_body_bytes=0)


def test_a_positive_cap_still_serves_events() -> None:
    # The guard must not turn the legitimate 413 path into a startup error.
    server = make_server(Monitor([], []), host="127.0.0.1", port=0, max_body_bytes=10)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/events",
            data=_EVENT,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=5)
        # 10-byte cap vs a ~150-byte body: still an over-cap rejection.
        assert excinfo.value.code == 413
    finally:
        server.shutdown()
        thread.join(timeout=2)
