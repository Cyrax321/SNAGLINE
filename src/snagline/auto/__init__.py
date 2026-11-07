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

"""Auto-instrumentation package (ATTACH_ANY_SYSTEM P0)."""

from snagline.auto.anthropic import instrument_anthropic
from snagline.auto.anthropic import wrap_client as wrap_anthropic
from snagline.auto.langchain import (
    instrument_langchain,
)
from snagline.auto.langchain import (
    wrap_client as wrap_langchain,
)
from snagline.auto.openai import instrument_openai, wrap_client

__all__ = [
    "instrument_openai",
    "wrap_client",
    "instrument_anthropic",
    "wrap_anthropic",
    "instrument_langchain",
    "wrap_langchain",
]
