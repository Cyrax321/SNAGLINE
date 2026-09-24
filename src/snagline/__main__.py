"""Package entry point so ``python -m snagline`` works.

Running a *package* with ``python -m`` requires a ``__main__`` submodule;
without this file ``python -m snagline`` fails with "No module named
snagline.__main__" even though the ``snagline`` console script and
``python -m snagline.cli`` both work. Delegates to the same ``main()``
the console script uses (see ``pyproject.toml`` ``[project.scripts]``).
"""

from __future__ import annotations

import sys

from snagline.cli import main

if __name__ == "__main__":
    sys.exit(main())
