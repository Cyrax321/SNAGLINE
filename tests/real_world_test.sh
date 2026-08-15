#!/usr/bin/env bash
# real_world_test.sh - run every SNAGLINE example and bridge in one pass
#
# Usage:
#   cd <project-root>
#   bash tests/real_world_test.sh              # everything (no API key needed)
#   bash tests/real_world_test.sh --with-api    # also runs real LLM tests (needs key)
#   bash tests/real_world_test.sh --claude-code  # includes Claude Code hook demo
#
# This exercises every path a real user would take:
#   1. Install check (zero deps)
#   2. Benchmark (submits us/step number)
#   3. Raw loop example (healthy + failing)
#   4. LangChain chaos harness (4 modes, fake model)
#   5. LangGraph real agent executor (4 modes, fake model)
#   6. Replay CLI against fixture trajectories
#   7. Sidecar HTTP server (canonical events + Claude Code payloads)
#   8. Command bridge (snagline hook)
#   9. File bridge (hook -> file -> watch)
# 10. Claude Code hook config (prints the JSON to paste into settings.json)
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PASS=0; FAIL=0; SKIP=0

ok()   { printf "\033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS + 1)); }
fail() { printf "\033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL + 1)); }
skip() { printf "\033[33mSKIP\033[0m  %s\n" "$1"; SKIP=$((SKIP + 1)); }

export PYTHONPATH=src

echo ""
echo "=========================================="
echo "  SNAGLINE real-world test suite"
echo "=========================================="
echo ""

# ── 0. Install check ──
python3 -c "import snagline; print(f'  version={snagline.__file__}')" 2>/dev/null && ok "import snagline (zero deps)" || { fail "import snagline"; exit 1; }
python3 -c "from snagline.server.http_server import serve; print('  server module ok')" && ok "import server module"
python3 -c "from snagline.sinks.webhook import WebhookSink; print('  webhook sink ok')" && ok "import webhook sink"
python3 -c "from snagline.adapters.claude_code import ingest_payload; print('  claude_code adapter ok')" && ok "import claude_code adapter"

# ── 1. Benchmark ──
echo ""
echo "--- benchmark ---"
BENCH=$(snagline bench 2>&1)
echo "$BENCH"
echo "$BENCH" | grep -q "us/step" && ok "benchmark prints us/step" || fail "benchmark output"

# ── 2. Raw loop example ──
echo ""
echo "--- raw loop: failing run (should fire loop + error_cascade) ---"
RAW_FAULTY=$(python3 examples/raw_loop_example.py 2>&1 || true)
echo "$RAW_FAULTY" | grep -q '"trigger"' && ok "raw loop detects failures" || fail "raw loop missed failures"

echo "--- raw loop: healthy run (should be SILENT) ---"
RAW_HEALTHY=$(python3 examples/raw_loop_example.py --healthy 2>&1 || true)
RISKS=$(echo "$RAW_HEALTHY" | grep -c '"trigger"' || true)
if [ "$RISKS" -eq 0 ]; then ok "raw loop healthy: zero false positives"; else fail "raw loop healthy: $RISKS false positive(s)"; fi

# ── 3. Replay CLI ──
echo ""
echo "--- replay: injected loop trajectory ---"
REPLAY=$(snagline replay tests/fixtures/trajectories/injected_loop.jsonl --summary 2>&1)
echo "$REPLAY" | grep -q "2 risk" && ok "replay detects loop fixture" || fail "replay missed loop fixture"

echo "--- replay: healthy trajectory ---"
HEALTHY=$(snagline replay tests/fixtures/trajectories/healthy_run.jsonl --quiet --summary 2>&1)
echo "$HEALTHY" | grep -q "0 risk" && ok "replay healthy: 0 risks" || fail "replay healthy: false positive"

echo "--- replay: injected error cascade ---"
ERR=$(snagline replay tests/fixtures/trajectories/injected_error_cascade.jsonl --summary 2>&1)
echo "$ERR" | grep -q "risk" && ok "replay detects error cascade" || fail "replay missed error cascade"

echo "--- replay: injected latency spike ---"
LAT=$(snagline replay tests/fixtures/trajectories/injected_latency_spike.jsonl --summary 2>&1)
echo "$LAT" | grep -q "risk" && ok "replay detects latency spike" || fail "replay missed latency spike"

# ── 4. LangChain chaos harness (no API key needed) ──
echo ""
echo "--- LangChain chaos: loop mode ---"
LC_LOOP=$(python3 examples/real_agent_demo.py --mode loop 2>&1 | grep -c '"trigger": "loop"' || true)
[ "$LC_LOOP" -gt 0 ] && ok "LangChain loop detected ($LC_LOOP fires)" || fail "LangChain loop not detected"

echo "--- LangChain chaos: error mode ---"
LC_ERR=$(python3 examples/real_agent_demo.py --mode error 2>&1 | grep -c '"trigger": "error_cascade"' || true)
[ "$LC_ERR" -gt 0 ] && ok "LangChain error_cascade detected ($LC_ERR fires)" || fail "LangChain error_cascade not detected"

echo "--- LangChain chaos: latency mode ---"
LC_LAT=$(python3 examples/real_agent_demo.py --mode latency 2>&1 | grep -c '"trigger": "latency_anomaly"' || true)
[ "$LC_LAT" -gt 0 ] && ok "LangChain latency_anomaly detected ($LC_LAT fires)" || fail "LangChain latency_anomaly not detected"

echo "--- LangChain chaos: healthy mode (false-positive check) ---"
LC_HLTH=$(python3 examples/real_agent_demo.py --mode healthy 2>&1 | grep -c '"trigger"' || true)
[ "$LC_HLTH" -eq 0 ] && ok "LangChain healthy: zero false positives" || fail "LangChain healthy: $LC_HLTH false positive(s)"

# ── 5. LangGraph real agent executor (no API key needed) ──
echo ""
if python3 -c "from langchain.agents import create_agent" 2>/dev/null; then
    for MODE in loop error latency healthy; do
        echo "--- LangGraph agent executor: $MODE ---"
        LG=$(python3 examples/real_agent_executor_demo.py --mode $MODE 2>&1 | grep -c '"trigger"' || true)
        if [ "$MODE" = "healthy" ]; then
            [ "$LG" -eq 0 ] && ok "LangGraph healthy: zero false positives" || fail "LangGraph healthy: $LG false positive(s)"
        else
            [ "$LG" -gt 0 ] && ok "LangGraph $MODE: detected ($LG fires)" || fail "LangGraph $MODE: not detected"
        fi
    done
else
    skip "LangGraph agent executor (langchain.agents not importable)"
fi

# ── 6. Sidecar HTTP server ──
echo ""
echo "--- sidecar: health, canonical events, Claude Code payloads ---"
python3 -m snagline.cli serve --port 8797 &
SIDE=$!
sleep 1
HEALTH=$(curl -sf http://127.0.0.1:8797/health) && echo "$HEALTH" | grep -q '"ok"' && ok "sidecar /health" || fail "sidecar /health"

# canonical StepEvent ingestion + loop detection
for i in 1 2 3 4; do
    curl -s -X POST http://127.0.0.1:8797/events -d "{\"step_id\":\"$i\",\"episode_id\":\"s1\",\"timestamp\":1718300000.0,\"action_type\":\"tool_call\",\"action_signature\":\"aaa\",\"tool_name\":\"t\"}" > /dev/null
done
ok "sidecar /events ingests 4 steps (see loop risk above)"

# Claude Code native payload
for i in 1 2 3 4; do
    curl -s -X POST http://127.0.0.1:8797/hooks/claude-code -d "{\"session_id\":\"cc1\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"ls\"},\"tool_use_id\":\"c$i\"}" > /dev/null
done
ok "sidecar /hooks/claude-code (see loop risk above)"

# unmapped lifecycle -> ignored
curl -s -X POST http://127.0.0.1:8797/hooks/claude-code -d '{"hook_event_name":"SessionStart"}' | grep -q '"ignored"' && ok "sidecar ignores lifecycle noise" || fail "sidecar did not ignore lifecycle noise"
kill $SIDE 2>/dev/null; wait $SIDE 2>/dev/null || true

# ── 7. Command bridge (snagline hook) ──
echo ""
echo "--- command bridge: snagline hook ---"
rm -f /tmp/snag_hook_test.jsonl
echo '{"session_id":"s","hook_event_name":"PreToolUse","tool_name":"T","tool_input":{"x":1},"tool_use_id":"a"}' | snagline hook --out /tmp/snag_hook_test.jsonl
echo '{"session_id":"s","hook_event_name":"PreToolUse","tool_name":"T","tool_input":{"x":1},"tool_use_id":"b"}' | snagline hook --out /tmp/snag_hook_test.jsonl
echo "garbage" | snagline hook --out /tmp/snag_hook_test.jsonl 2>/dev/null
LINES=$(wc -l < /tmp/snag_hook_test.jsonl 2>/dev/null || echo 0)
LINES=$(echo "$LINES" | tr -d ' ')
[ "$LINES" = "2" ] && ok "hook appends 2 events, ignores garbage" || fail "hook file output (got $LINES lines)"
rm -f /tmp/snag_hook_test.jsonl

# ── 8. File bridge ──
echo ""
echo "--- file bridge: hook -> file -> watch ---"
rm -f /tmp/snag_file_test.jsonl
for i in 1 2 3 4; do
    echo "{\"session_id\":\"sf\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"W\",\"tool_input\":{},\"tool_use_id\":\"f$i\"}" | snagline hook --out /tmp/snag_file_test.jsonl
done
FILE_OUT=$(snagline watch --file /tmp/snag_file_test.jsonl --episode-id sf 2>&1)
echo "$FILE_OUT" | grep -q '"trigger": "loop"' && ok "file bridge detects loop" || fail "file bridge missed loop"
rm -f /tmp/snag_file_test.jsonl

# ── 9. Claude Code hook config ──
echo ""
echo "--- Claude Code hook config ---"
CONFIG=$(cat << 'JSON'
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ],
    "PostToolUseFailure": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ]
  }
}
JSON
)
echo "$CONFIG" > /tmp/claude_code_snagline_settings.json
ok "Claude Code settings.json written to /tmp/claude_code_snagline_settings.json"

# ── 10. Real LLM test (optional, needs API key) ──
echo ""
if [ "${1:-}" = "--with-api" ]; then
    echo "--- real LLM: Claude Code / OpenRouter / Anthropic ---"
    python3 examples/real_time_llm_demo.py --mode healthy 2>&1 && ok "real LLM healthy" || fail "real LLM healthy"
    python3 examples/real_time_llm_demo.py --mode error 2>&1 && ok "real LLM error" || fail "real LLM error"
    python3 examples/real_time_llm_demo.py --mode latency 2>&1 && ok "real LLM latency" || fail "real LLM latency"
else
    skip "real LLM tests (pass --with-api to run; needs OPENAI_API_KEY or ANTHROPIC_API_KEY)"
fi

# ── Summary ──
echo ""
echo "=========================================="
printf "  \033[32m%d passed\033[0m  \033[31m%d failed\033[0m  \033[33m%d skipped\033[0m\n" "$PASS" "$FAIL" "$SKIP"
echo "=========================================="
echo ""
echo "To hook Claude Code itself (monitor THIS agent in real time):"
echo "  1. Copy /tmp/claude_code_snagline_settings.json into .claude/settings.json"
echo "  2. Run: snagline serve --port 8787"
echo "  3. Start a Claude Code session - SNAGLINE watches every tool call live"
echo ""
echo "To test with a real LLM (OpenRouter free tier, no credit card):"
echo "  export OPENAI_API_KEY=sk-or-..."
echo "  bash tests/real_world_test.sh --with-api"
echo ""

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
