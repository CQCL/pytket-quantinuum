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

from .api_wrappers import (
    QuantinuumAPI as QuantinuumAPI,
)
from .api_wrappers import (
    QuantinuumAPIOffline as QuantinuumAPIOffline,
)
from .leakage_gadget import (
    prune_shots_detected_as_leaky as prune_shots_detected_as_leaky,
)
from .quantinuum import (
    Language as Language,
)
from .quantinuum import (
    QuantinuumBackend as QuantinuumBackend,
)
from .quantinuum import (
    QuantinuumBackendCompilationConfig as QuantinuumBackendCompilationConfig,
)
from .quantinuum import (
    have_pecos as have_pecos,
)
