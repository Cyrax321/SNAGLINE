"""Opening user-named input files with one diagnostic shape (issue #595).

Kept dependency-free and below every CLI command so ``cli`` and ``baseline``
share it without importing each other. Three concerns live here:

* ``_BadInputFile`` subclasses ``OSError`` so library callers that already catch
  ``OSError`` from a read keep working, while the CLI catches the subclass to
  turn the failure into a one-line diagnostic instead of a traceback.
* ``open(..., encoding="utf-8")`` decodes lazily, so a non-UTF-8 trajectory
  raises ``UnicodeDecodeError`` from the *iterator*, not from ``open``. That
  subclasses ``ValueError``, not ``OSError``, so a plain ``except OSError``
  guard never fires and the command tracebacks part way through the file.
* ``_DecodeGuardedFile`` exists to close that gap for every read shape the CLI
  uses (``for line in fh``, ``fh.readline()``, ``json.load(fh)``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import IO, TypeVar

_T = TypeVar("_T")


class _BadInputFile(OSError):
    """A user-named input file could not be opened or decoded (issues #428,
    #595).

    Subclasses ``OSError`` so library callers that already catch ``OSError``
    from a read keep working; the CLI catches this subclass to turn the failure
    into a one-line diagnostic instead of a traceback.
    """


def open_input(path: str) -> _DecodeGuardedFile:
    """Open a named input file, or fail naming it (issue #428).

    ``open()``'s own message already names the path, but the CLI wants the
    command in the diagnostic too, and a raw ``FileNotFoundError`` surfacing
    out of ``main()`` reads as a crash of the tool itself. The result is the
    guard wrapper rather than the raw ``IO[str]`` so every read shape the CLI
    uses reports a decode failure the same way a missing file already does.
    """
    try:
        fh = open(path, encoding="utf-8")  # noqa: SIM115 - the handle is
        # returned wrapped in _DecodeGuardedFile, so a with-block here would
        # close it before the caller could read it.
    except OSError as exc:
        raise _BadInputFile(f"cannot open {path}: {exc}") from exc
    return _DecodeGuardedFile(fh, path)


class _DecodeGuardedFile:
    """An open text input whose decode failures read as a bad input file.

    Wraps rather than subclasses ``TextIOWrapper`` so every call site keeps its
    existing ``with`` / iteration shape and the diagnostic names the file the
    same way a missing file already does (issue #595).
    """

    def __init__(self, fh: IO[str], path: str) -> None:
        self._fh = fh
        self._path = path

    def __enter__(self) -> _DecodeGuardedFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _guard(self, op: Callable[[], _T]) -> _T:
        try:
            return op()
        except UnicodeDecodeError as exc:
            raise _BadInputFile(f"cannot decode {self._path} as utf-8: {exc}") from exc

    def __iter__(self) -> Iterator[str]:
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def readline(self, size: int = -1) -> str:
        return self._guard(lambda: self._fh.readline(size))

    def readlines(self, hint: int = -1) -> list[str]:
        return self._guard(lambda: self._fh.readlines(hint))

    def read(self, size: int = -1) -> str:
        return self._guard(lambda: self._fh.read(size))

    def close(self) -> None:
        self._fh.close()
