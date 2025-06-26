# Copyright Quantinuum
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

"""Backends for processing pytket circuits with Quantinuum devices"""

# _metadata.py is copied to the folder after installation.
from ._metadata import __extension_name__, __extension_version__
from .backends import (
    H1,
    H2,
    Language,
    QuantinuumAPI,
    QuantinuumAPIOffline,
    QuantinuumBackend,
    QuantinuumBackendCompilationConfig,
    QuantinuumBackendData,
    have_pecos,
    prune_shots_detected_as_leaky,
)
