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

from collections import Counter
import os
from pathlib import Path
import numpy as np
import pytest
from pytket.circuit import (
    Circuit,
    Qubit,
    Bit,
    reg_eq,
    reg_neq,
    reg_lt,
    reg_gt,
    reg_leq,
    reg_geq,
    if_not_bit,
)
from pytket.extensions.quantinuum import QuantinuumBackend, have_pecos
from pytket.wasm import WasmFileHandler

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Quantinuum username)"
)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_local_emulator(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa
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
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_circuit_with_conditional(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa
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
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_results_order(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa
    c0 = Circuit(2).X(0).measure_all()
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert counts == Counter({(1, 0): 10})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_multireg(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa
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


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_setbits(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit(1, 3)
    c.H(0)
    c.Measure(0, 0)
    c.add_c_setbits([True, False, True], c.bits)
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots).get_counts()
    assert counts == Counter({(1, 0, 1): n_shots})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_classical_0(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore

    c.add_c_setreg(23, a)
    c.add_c_copyreg(a, b)

    c.add_classicalexpbox_register(a + b, d.to_list())
    c.add_classicalexpbox_register(a - b, d.to_list())
    c.add_classicalexpbox_register(a * b * d, d.to_list())
    c.add_classicalexpbox_register(a << 1, a.to_list())
    c.add_classicalexpbox_register(a >> 1, b.to_list())

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.X(0, condition=reg_eq(a & b, 1))
    c.X(0, condition=reg_eq(a | b, 1))

    c.X(0, condition=a[0])
    c.X(0, condition=reg_neq(a, 1))
    c.X(0, condition=if_not_bit(a[0]))
    c.X(0, condition=reg_gt(a, 1))
    c.X(0, condition=reg_lt(a, 1))
    c.X(0, condition=reg_geq(a, 1))
    c.X(0, condition=reg_leq(a, 1))
    c.Phase(0, condition=a[0])
    c.Measure(Qubit(0), d[0])

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    counts = backend.run_circuit(c, n_shots=10).get_counts()
    assert len(counts.values()) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_classical_1(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore

    c.add_c_setreg(23, a)
    c.add_c_copyreg(a, b)

    c.add_classicalexpbox_register(a + b, d.to_list())
    c.add_classicalexpbox_register(a - b, d.to_list())
    c.add_classicalexpbox_register(a * b * d, d.to_list())
    c.add_classicalexpbox_register(a << 1, a.to_list())
    c.add_classicalexpbox_register(a >> 1, b.to_list())

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.Measure(Qubit(0), d[0])

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    counts = backend.run_circuit(c, n_shots=10).get_counts()
    assert len(counts.values()) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_classical_2(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 2)
    b = circ.add_c_register("b", 2)
    c = circ.add_c_register("c", 1)
    expr = a[0] ^ b[0]
    circ.add_classicalexpbox_bit(expr, [c[0]])
    circ.X(0)
    circ.Measure(Qubit(0), a[1])
    backend = authenticated_quum_backend_qa
    cc = backend.get_compiled_circuit(circ)
    counts = backend.run_circuit(cc, n_shots=10).get_counts()
    assert len(counts.keys()) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_classical_3(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 4)
    b = circ.add_c_register("b", 4)
    c = circ.add_c_register("c", 4)

    circ.add_c_setreg(3, a)
    circ.add_c_copyreg(a, b)

    circ.add_classicalexpbox_register(a - b, c.to_list())
    circ.add_classicalexpbox_register(a << 1, a.to_list())

    circ.X(0)
    circ.Measure(Qubit(0), a[3])

    backend = authenticated_quum_backend_qa

    cc = backend.get_compiled_circuit(circ)
    counts = backend.run_circuit(cc, n_shots=10).get_counts()
    assert len(counts.keys()) == 1
    result = list(counts.keys())[0]
    assert result == (0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_wasm(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])

    b = authenticated_quum_backend_qa

    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, wasm_file_handler=wasfile, n_shots=n_shots).get_counts()  # type: ignore
    assert counts == Counter({(1, 0, 0, 0, 0, 0, 0, 0): n_shots})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
# Same as test_wasm_collatz() in backend_test.py but run on local emulators.
def test_wasm_collatz(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    wasfile = WasmFileHandler(
        str(Path(__file__).parent.parent / "wasm" / "collatz.wasm")
    )
    c = Circuit(8)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 8)

    # Use Hadamards to set "a" register to a random value.
    for i in range(8):
        c.H(i)
        c.Measure(Qubit(i), Bit("a", i))
    # Compute the value of the Collatz function on this value.
    c.add_wasm_to_reg("collatz", wasfile, [a], [b])

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(c, n_shots=10, wasm_file_handler=wasfile)  # type: ignore

    r = backend.get_result(h)
    shots = r.get_shots()

    def to_int(C: np.ndarray) -> int:
        assert len(C) == 8
        return sum(pow(2, i) * C[i] for i in range(8))

    def collatz(n: int) -> int:
        if n == 0:
            return 0
        m = 0
        while n != 1:
            n = (3 * n + 1) // 2 if n % 2 == 1 else n // 2
            m += 1
        return m

    for shot in shots:
        n, m = to_int(shot[:8]), to_int(shot[8:16])
        assert collatz(n) == m


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_midcircuit_measurement_and_reset(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    c = Circuit(1, 4)
    c.X(0)
    c.Measure(0, 0)
    c.Reset(0)
    c.Measure(0, 1)
    c.X(0)
    c.Measure(0, 2)
    c.Measure(0, 3)

    b = authenticated_quum_backend_qa

    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots).get_counts()
    assert counts == Counter({(1, 0, 1, 1): n_shots})


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_LOCAL_SIMULATOR_NAMES],  # type: ignore
    indirect=True,
)
def test_cbits(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 2)
    b = circ.add_c_register("b", 2)

    circ.add_c_setreg(3, a)
    circ.add_c_copyreg(a, b)
    circ.X(0)
    circ.Measure(Qubit(0), a[0])

    backend = authenticated_quum_backend_qa

    cc = backend.get_compiled_circuit(circ)
    r = backend.run_circuit(cc, n_shots=1)
    counts = r.get_counts(cbits=list(a))
    assert counts == Counter({(1, 1): 1})
