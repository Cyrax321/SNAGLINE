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

"""Optional ``drift`` extra: semantic goal-drift detection (issue #81).

Provides ``pip install snagline[drift]`` (sentence-transformers). This
subpackage is never imported by core code paths unless
``Config.semantic_drift_enabled`` is set; ``import snagline`` works with
nothing but the standard library (project.md section 1.1). Even this module
itself imports cleanly without the extra installed: the heavy dependency is
touched only inside the lazy model loader, and any failure there leaves the
detector inert (caught, logged, ignored) instead of raising into the host.
"""
