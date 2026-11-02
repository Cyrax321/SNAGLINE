# Contributing to SNAGLINE

Short and practical: the path from clone to green suite, then the conventions
the maintainers hold. For the design contracts behind detectors, adapters, and
bridges, read [docs/DETECTOR_GUIDE.md](docs/DETECTOR_GUIDE.md),
[docs/ADAPTER_GUIDE.md](docs/ADAPTER_GUIDE.md), and
[docs/FRAMEWORK_BRIDGES.md](docs/FRAMEWORK_BRIDGES.md) — this file does not
repeat them.

## Setup

The core has zero runtime dependencies; the `dev` extra carries the pinned CI
tooling (pytest, pytest-cov, ruff, mypy — the pin versions match CI).

```bash
git clone https://github.com/Cyrax321/SNAGLINE.git
cd SNAGLINE
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Keep the venv active for everything below, including the smoke script, which
invokes the `snagline` console script rather than `python -m`.

## The gates

CI runs six jobs: a test matrix (OS x Python), the `ml` and `continuum` extra
legs, an accuracy gate, a real-world smoke job, and a version-tag check. Local
equivalents, in the order worth running them:

```bash
ruff check src tests && ruff format --check src tests
mypy src tests
python -m pytest tests/ -q
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format table
bash tests/real_world_test.sh
```

Real status lines on master (`9dbed7f`, Python 3.12, macOS). Run the commands
yourself — file counts, timings, and test counts all vary by branch, platform,
and installed extras, so the convention everywhere in this repo is to quote
the command, not the result:

```
$ ruff check src tests
All checks passed!
$ ruff format --check src tests
... files already formatted
$ mypy src tests
Success: no issues found in <tree> source files
$ python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format table
...per-trigger precision/recall/F1 table...
macro-F1: 0.963
episodes: 116 (72 labeled, 44 healthy controls)
healthy controls that fired: 0
$ bash tests/real_world_test.sh
...one PASS/FAIL section per example and bridge, all green...
```

Three notes that cost contributors CI rounds:

- **`mypy` type-checks `tests` too.** `mypy src` alone passes locally and
  fails CI.
- **The accuracy gate exits non-zero if any healthy control episode fires a
  risk.** A detector change that catches more real failures can also start
  false-positiving; run it the same way CI does before claiming clean.
- **Optional extras un-skip their own test legs**: `pip install ".[langchain]"`
  for the LangChain integration tests, `".[ml]"` for the ESN ensemble leg,
  `".[langgraph]"` / `".[openai]"` / `".[anthropic]"` for their adapters. CI
  covers the `langchain` and `ml` legs explicitly; the rest skip gracefully,
  so a green local run is not proof an adapter test passed.

## Writing tests

- A bug fix ships with the regression test that reproduces it, plus a
  `CHANGELOG.md` entry under `[Unreleased]` naming the issue number.
- Tests live in `tests/`: detector suites under `tests/detectors/`, adapter,
  config, server, and sink suites in their subdirectories, everything else at
  the top level. Name a new file after what it pins.
- Type the tests (`-> None`, annotated fixtures). `mypy src tests` checks
  them, so an untyped test is a CI failure, not a style nit.
- Never quote a test count in prose. `tests/test_no_hardcoded_test_counts.py`
  scans `README.md` and `project.md` for them and fails the build; the
  convention this repo holds is wider than what the guard can check, so apply
  it here too — the CI badge is the only place a count belongs.

## Issues

The bar is verification: name the `file:line`, the command that reproduces it,
and that command's **real output on the current master**, and state the master
commit you verified against. The issue template asks for exactly this.
"X does not work" without a reproduction gets closed.

Search open issues *and* open pull requests before filing. SNAGLINE has a
long history of fixes landing on `master` while the issue stays open: batch
fix PRs cite issue numbers in their title as a **scope list, never a closing
keyword**, so the issues never auto-close (#576). An issue that looks open may
already be fixed — check a fresh `master` checkout before concluding nobody
has looked at it.

## Good first issues

Beginner-friendly work carries the `good first issue` label on the
[issue tracker](https://github.com/Cyrax321/SNAGLINE/issues). Only a
maintainer can apply it; if you file an issue you think is newcomer-sized,
list suggested labels in its footer instead of waiting for triage rights.

## Pull requests

Fill in the whole template:

- **Summary of Changes** and **Justification** — what changed, and which use
  case it serves. For a bug, that includes the reproduction evidence.
- **Unit Tests** — the new/updated cases, and what each pins.
- **Execution Tests** — end-to-end evidence, not "tests pass": attach the
  monitor to an example under `examples/` or a real run and report what
  SNAGLINE observed.
- **Benchmark** — only when the change affects detection latency or overhead,
  with `benchmarks/overhead_benchmark.py` numbers before and after.

Reference one issue per PR with `Closes #N`. A batch PR may list issue numbers
in its title as scope, but must not use a closing keyword: a closing keyword
auto-closes every issue in the list when the batch merges, whether or not it
was actually fixed (#576).

Target `master` and keep the diff focused — a PR that mixes a bug fix with
unrelated cleanup is harder to review and more likely to conflict.
