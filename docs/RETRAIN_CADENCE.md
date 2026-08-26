# Scheduled baseline retrain cadence (issue #102)

The `goal_drift` detector compares live traffic against a persisted healthy
reference (`BaselineProfile`). Real deployments drift: tool latencies change,
new tools ship, error budgets move. A baseline fitted once and never touched
slowly turns into a false-positive generator. The retrain contract closes that
loop: refit from the newest healthy-run window on a host-owned schedule, bump
the versioned store atomically, and warn when the active baseline has gone
stale anyway.

Everything here is stdlib-only and offline: no new dependencies, no network,
no content retention (see the privacy note at the end).

## The command

```
snagline baseline retrain --store-dir ROOT [--tenant T] [--deployment D]
    (--jsonl FILE | --windows-dir DIR) [--max-age SECONDS] [--max-versions N]
```

Behavior:

- Refits a fresh profile from the given JSONL window (`--jsonl`) or from the
  newest `*.jsonl` file in a directory of rotated windows (`--windows-dir`,
  picked by modification time, non-recursive).
- Stores the result as a NEW timestamped version in the `BaselineStore` and
  flips the `latest.json` pointer atomically (see below). The previous version
  stays loadable for rollback.
- Prints the stored version id plus a small structural summary (tool count,
  step count). No content is echoed.
- Exit codes: `0` success, `2` usage error (missing `--store-dir`, wrong
  combination of window inputs), `3` input failure (no window found, window
  unreadable). Malformed lines inside a window are skipped fail-open during
  the fit, matching replay behavior.

The literal keyword `retrain` replaces the usual trajectory positional. The
plain form (`snagline baseline run.jsonl --output baseline.json`) is unchanged.

## Cron example

Refit nightly from the newest rotated window; keep a staleness guard so a
silently broken pipeline pages someone instead of drifting unnoticed:

```cron
# m h dom mon dow  command
17 3 * * *  /opt/snagline/.venv/bin/snagline baseline retrain --store-dir /var/lib/snagline/baselines --tenant acme --deployment prod --windows-dir /var/log/agent/windows --max-age 172800 >> /var/log/snagline-retrain.log 2>&1
```

Notes on this line:

- `17 3`: an off-the-hour minute avoids the midnight cron thundering herd.
- Absolute paths everywhere: cron's environment has no venv on PATH.
- `--max-age 172800` (48h): if the ACTIVE baseline was fitted more than two
  days ago, retrain prints a `WARNING ... consider tightening the retrain
  cadence` line to stderr before doing its work. With the redirect above that
  warning lands in `/var/log/snagline-retrain.log`; point a log alert at the
  word `WARNING` for that file.
- The exit code is still `0` when retrain succeeds despite the warning. Exit
  `2`/`3` mean nothing was stored, which is what you want cron mail or a
  `systemd` `on-failure` handler to catch.

## systemd timer example

Equivalent to the cron line, with catch-up on missed schedules:

```ini
# /etc/systemd/system/snagline-retrain.service
[Unit]
Description=SNAGLINE baseline retrain from newest JSONL window

[Service]
Type=oneshot
ExecStart=/opt/snagline/.venv/bin/snagline baseline retrain --store-dir /var/lib/snagline/baselines --tenant acme --deployment prod --windows-dir /var/log/agent/windows --max-age 172800
```

```ini
# /etc/systemd/system/snagline-retrain.timer
[Unit]
Description=Run SNAGLINE baseline retrain nightly

[Timer]
OnCalendar=*-*-* 03:17:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with `systemctl enable --now snagline-retrain.timer`. `Persistent=true`
runs the service once on boot if the last scheduled run was missed (laptop or
fleet reboot friendly).

## Why the version bump is atomic

`BaselineStore.save()` writes each JSON payload to a sibling temp file, fsyncs
it, then renames it over the target with `os.replace` (atomic on POSIX and
Windows). Order matters: the history entry `versions/<id>.json` lands first,
then the `latest.json` pointer flips. A crash between the two writes leaves
the active baseline pointing at the previous complete version; readers of
`latest.json` never observe torn or half-written JSON, only old-or-new. The
orphaned history entry is harmless and remains loadable. This guarantee is
what makes running retrain against a live store safe without locks.

Retention is bounded: versions beyond `--max-versions` (default 10) are pruned
oldest-first, so the store grows by one small JSON file per retrain, not
forever. Rollback is reading an older version out of
`<store-dir>/<tenant>/<deployment>/versions/`.

## Retrain is not a mid-episode reset for goal_drift

Read this before wiring retrain into a fleet of long-running agents:

- A retrain updates files under the store root. It does NOT hot-swap the
  baseline object held by an already-running monitor. Hosts should reload at
  an episode boundary: rebuild the monitor config with
  `BaselineStore.load(...)` (the pattern shown in the README) so future
  episodes compare against the fresh profile.
- Even with a new reference loaded mid-flight, the goal_drift detector keeps
  its per-episode live accumulators and its fired-once flag. Retraining does
  not un-fire risks already raised and does not wipe partial episode state;
  scores for the in-flight episode simply continue against the new reference.
  For a clean-slate comparison, start a new episode rather than expecting the
  retrain to reset detector state.

## Privacy reminder

Baselines contain structure only: per-tool step counts, latency
sums/means/std/min/max, and error counts. Fitted windows are read line by line
and reduced to those aggregates; no prompt or response content ever reaches
`baseline.json`, the store, or the retrain log output. Rotate window files
under your normal log-retention policy; the baseline store itself never needs
to hold content.
