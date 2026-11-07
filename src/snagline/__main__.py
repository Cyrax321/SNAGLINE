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

"""Package entry point so ``python -m snagline`` works (issue #487).

The console-script entry point (``snagline = "snagline.cli:main"``) covers the
bare ``snagline`` command, and ``cli.py`` ends with an ``if __name__ ==
"__main__"`` guard -- but that guard only fires for ``python -m snagline.cli``.
Running a *package* with ``-m`` requires a ``__main__`` submodule, so without
this file ``python -m snagline`` fails outright with "cannot be directly
executed". This mirrors the convenience of ``python -m pip`` / ``python -m
pytest`` for environments where the Scripts directory is not on PATH.
"""

from __future__ import annotations

import sys

from snagline.cli import main

if __name__ == "__main__":
    sys.exit(main())
