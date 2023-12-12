# Copyright 2020-2023 Cambridge Quantum Computing
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

from collections import Counter
import os
import pytest
from pytket.circuit import Circuit, Qubit, Bit
from pytket.extensions.quantinuum import QuantinuumBackend, have_pecos

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Quantinuum username)"
)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize("device_name", pytest.ALL_LOCAL_SIMULATOR_NAMES)
def test_local_emulator(device_name: str) -> None:
    b = QuantinuumBackend(device_name)
    assert b.is_local_emulator
    c0 = Circuit(2).X(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c0)
    assert b.cost(c, n_shots=10) == 0.0
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert counts == Counter({(1, 1): 10})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize("device_name", pytest.ALL_LOCAL_SIMULATOR_NAMES)
def test_circuit_with_conditional(device_name: str) -> None:
    b = QuantinuumBackend(device_name)
    c0 = Circuit(2, 2).H(0)
    c0.Measure(0, 0)
    c0.X(1, condition_bits=[0], condition_value=1)
    c0.Measure(1, 1)
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert sum(counts.values()) == 10
    assert all(v0 == v1 for v0, v1 in counts.keys())


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize("device_name", pytest.ALL_LOCAL_SIMULATOR_NAMES)
def test_results_order(device_name: str) -> None:
    b = QuantinuumBackend(device_name)
    c0 = Circuit(2).X(0).measure_all()
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert counts == Counter({(1, 0): 10})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize("device_name", pytest.ALL_LOCAL_SIMULATOR_NAMES)
def test_multireg(device_name: str) -> None:
    b = QuantinuumBackend(device_name)
    c = Circuit()
    q1 = Qubit("q1", 0)
    q2 = Qubit("q2", 0)
    c1 = Bit("c1", 0)
    c2 = Bit("c2", 0)
    for q in (q1, q2):
        c.add_qubit(q)
    for cb in (c1, c2):
        c.add_bit(cb)
    c.H(q1)
    c.CX(q1, q2)
    c.Measure(q1, c1)
    c.Measure(q2, c2)
    c = b.get_compiled_circuit(c)

    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots).get_counts()
    assert sum(counts.values()) == 10
    assert all(v0 == v1 for v0, v1 in counts.keys())
