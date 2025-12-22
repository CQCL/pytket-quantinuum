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

import pytest

from pytket.circuit import Circuit
from pytket.extensions.quantinuum import (
    H2,
    QuantinuumBackend,
    QuantinuumBackendData,
    have_pecos,
)


def test_backend_data() -> None:
    for data in [H2]:
        b = QuantinuumBackend("test", data=data)
        c = Circuit(2).H(0).CX(0, 1).measure_all()
        p = b.default_compilation_pass(optimisation_level=2)
        p.apply(c)
        assert (
            c
            == Circuit(2)
            .PhasedX(3.5, 0.5, 0)
            .PhasedX(2.5, 0.5, 1)
            .ZZPhase(0.5, 0, 1)
            .PhasedX(0.5, 0.0, 1)
            .add_phase(1.75)
            .measure_all()
        )


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_backend_data_local_emulator() -> None:
    data = QuantinuumBackendData(n_qubits=20, n_cl_reg=10, local_emulator=True)
    b = QuantinuumBackend("test", data=data)
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c1 = b.get_compiled_circuit(c)
    h = b.process_circuit(c1, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert all(x0 == x1 for x0, x1 in counts)
    assert counts.total() == 10
