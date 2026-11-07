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

"""Adapter package -- the only framework-coupled code (project.md §1.3).

Each adapter turns a framework-specific event into a ``StepEvent`` and calls
``monitor.ingest``. The ``raw`` adapter is stdlib-only and always available;
framework adapters are optional extras and are duck-typed, so importing them
never requires the framework to be installed.
"""

from snagline.adapters.autogen import SnaglineAutogenHandler, run_and_monitor
from snagline.adapters.crewai import observe_crewai_step, snagline_step_callback
from snagline.adapters.raw import watch

__all__ = [
    "watch",
    "SnaglineAutogenHandler",
    "run_and_monitor",
    "snagline_step_callback",
    "observe_crewai_step",
]
