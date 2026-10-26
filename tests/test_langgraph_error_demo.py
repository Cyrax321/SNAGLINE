"""The LangGraph executor demo's error mode must actually fire error_cascade.

``examples/real_agent_executor_demo.py --mode error`` announced "expected:
error_cascade detector" and then crashed with an unhandled ``RuntimeError``,
emitting zero risks (issue #614): LangGraph's ``ToolNode`` re-raises any plain
tool exception straight out of the graph, before ``on_tool_error`` is
dispatched to the handler, so no error event ever reached the Monitor. The
failure is asserted by the smoke script's section 5
(``tests/real_world_test.sh``), which is precisely why only this mode was red.

The demo now surfaces each failure through ``ToolErrorMiddleware``: the run
survives the first failing tool call, the agent sees each error as a
``ToolMessage``, and ``on_tool_error`` still fires per call (LangChain
dispatches it inside ``BaseTool.run``, before the ToolNode sees the
exception) -- so consecutive failures accumulate and the cascade trips.

Skip-guarded: the demo drives the real ``langchain``/``langgraph``. The CI
matrix installs only ``langchain-core`` for the adapter tests and the smoke
job is stdlib-only, so these run wherever the framework is installed -- the
same skip-guarded arrangement as the real-dependency legs elsewhere.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DEMO = REPO / "examples" / "real_agent_executor_demo.py"


def _demo_available() -> None:
    pytest.importorskip("langchain")
    pytest.importorskip("langgraph")
    # ToolErrorMiddleware arrives with langchain 1.x's agent middleware; an
    # older langchain that can build the agent but not this demo's wiring is
    # a version question, not something this test can fake.
    try:
        from langchain.agents.middleware import ToolErrorMiddleware  # noqa: F401
    except ImportError:
        pytest.skip("langchain.agents.middleware.ToolErrorMiddleware unavailable")


def _run_demo(mode: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # The demo runs against the checkout's src, exactly as the smoke script
    # runs it (PYTHONPATH=src), so the subprocess cannot silently pick up a
    # different installed snagline.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, str(DEMO), "--mode", mode],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_error_mode_fires_error_cascade_without_crashing() -> None:
    _demo_available()
    result = _run_demo("error")
    # The pre-fix run died with the tool's RuntimeError out of agent.invoke.
    assert result.returncode == 0, result.stderr
    assert '"trigger": "error_cascade"' in (result.stdout + result.stderr), (
        "the error-mode demo reports no error_cascade risk, so the run either "
        "crashed before accumulating failures or stopped emitting them"
    )
    assert "Traceback" not in (result.stdout + result.stderr), result.stderr[-800:]


def test_error_mode_is_still_an_error_for_the_monitor() -> None:
    """The surfaced failures are real events, not silenced ones.

    The middleware hands the failure to the agent as a message; that is
    invisible to a reader of the transcript, so pin that the detector sees
    *error* events rather than the run merely completing.
    """
    _demo_available()
    result = _run_demo("error")
    assert result.returncode == 0, result.stderr
    body = result.stdout + result.stderr
    assert "error_cascade" in body
    # Pre-#614 the mode printed the intro but never a risk line; both the
    # crash path and the all-fixed-up message path must not regress into it.
    assert body.count('"trigger"') >= 1


def test_healthy_mode_stays_silent_with_middleware_attached() -> None:
    """The error-mode wiring is inert for a healthy run (false-positive check).

    ``ToolErrorMiddleware`` is attached for all four modes; a healthy run
    never triggers it, but the demo's healthy mode is the demo's only
    false-positive guard, so keep it pinned here too.
    """
    _demo_available()
    result = _run_demo("healthy")
    assert result.returncode == 0, result.stderr
    body = result.stdout + result.stderr
    assert '"trigger"' not in body, f"healthy mode false-positived:\n{body[-800:]}"
    assert "Traceback" not in body, result.stderr[-800:]


def test_loop_mode_still_fires() -> None:
    """The error-mode wiring did not change the other modes' detections."""
    _demo_available()
    result = _run_demo("loop")
    assert result.returncode == 0, result.stderr
    assert '"trigger"' in (result.stdout + result.stderr)
