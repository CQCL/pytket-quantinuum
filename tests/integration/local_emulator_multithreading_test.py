# Copyright 2020-2024 Cambridge Quantum Computing
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

import os
import pytest
from pytket.circuit import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend, have_pecos

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Quantinuum username)"
)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_multithreading(authenticated_quum_backend: QuantinuumBackend) -> None:
    b = authenticated_quum_backend
    c0 = Circuit(2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10, multithreading=True)
    r = b.get_result(h)
    counts = r.get_counts()
    assert all(x0 == x1 for x0, x1 in counts.keys())
