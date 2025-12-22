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

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pytket.circuit import (
    Bit,
    Circuit,
    Qubit,
    if_not_bit,
    reg_eq,
    reg_geq,
    reg_gt,
    reg_leq,
    reg_lt,
    reg_neq,
)
from pytket.circuit.clexpr import wired_clexpr_from_logic_exp
from pytket.extensions.quantinuum import QuantinuumBackend, have_pecos
from pytket.wasm import WasmFileHandler


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_local_emulator() -> None:
    b = QuantinuumBackend("H2-1LE")
    assert b.is_local_emulator
    c0 = Circuit(2).X(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert counts == Counter({(1, 1): 10})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_circuit_with_conditional() -> None:
    b = QuantinuumBackend("H2-1LE")
    c0 = Circuit(2, 2).H(0)
    c0.Measure(0, 0)
    c0.X(1, condition_bits=[0], condition_value=1)
    c0.Measure(1, 1)
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert sum(counts.values()) == 10
    assert all(v0 == v1 for v0, v1 in counts)


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_results_order() -> None:
    b = QuantinuumBackend("H2-1LE")
    c0 = Circuit(2).X(0).measure_all()
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10)
    r = b.get_result(h)
    counts = r.get_counts()
    assert counts == Counter({(1, 0): 10})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_multireg() -> None:
    b = QuantinuumBackend("H2-1LE")
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
    assert all(v0 == v1 for v0, v1 in counts)


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_setbits() -> None:
    b = QuantinuumBackend("H2-1LE")
    c = Circuit(1, 3)
    c.H(0)
    c.Measure(0, 0)
    c.add_c_setbits([True, False, True], c.bits)
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots).get_counts()
    assert counts == Counter({(1, 0, 1): n_shots})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_0() -> None:
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore

    c.add_c_setreg(23, a)
    c.add_c_copyreg(a, b)

    c.add_clexpr(*wired_clexpr_from_logic_exp(a + b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a - b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a * b * d, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a << 1, a.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a >> 1, b.to_list()))

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

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    counts = backend.run_circuit(c, n_shots=10).get_counts()
    assert len(counts.values()) == 1


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_1() -> None:
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore

    c.add_c_setreg(23, a)
    c.add_c_copyreg(a, b)

    c.add_clexpr(*wired_clexpr_from_logic_exp(a + b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a - b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a * b * d, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a << 1, a.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a >> 1, b.to_list()))

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.Measure(Qubit(0), d[0])

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    counts = backend.run_circuit(c, n_shots=10).get_counts()
    assert len(counts.values()) == 1


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_2() -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 2)
    b = circ.add_c_register("b", 2)
    c = circ.add_c_register("c", 1)
    expr = a[0] ^ b[0]
    circ.add_clexpr(*wired_clexpr_from_logic_exp(expr, [c[0]]))
    circ.X(0)
    circ.Measure(Qubit(0), a[1])
    backend = QuantinuumBackend("H2-1LE")
    cc = backend.get_compiled_circuit(circ)
    counts = backend.run_circuit(cc, n_shots=10).get_counts()
    assert len(counts.keys()) == 1


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_3() -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 4)
    b = circ.add_c_register("b", 4)
    c = circ.add_c_register("c", 4)

    circ.add_c_setreg(3, a)
    circ.add_c_copyreg(a, b)

    circ.add_clexpr(*wired_clexpr_from_logic_exp(a - b, c.to_list()))
    circ.add_clexpr(*wired_clexpr_from_logic_exp(a << 1, a.to_list()))

    circ.X(0)
    circ.Measure(Qubit(0), a[3])

    backend = QuantinuumBackend("H2-1LE")

    cc = backend.get_compiled_circuit(circ)
    counts = backend.run_circuit(cc, n_shots=10).get_counts()
    assert len(counts.keys()) == 1
    result = list(counts.keys())[0]  # noqa: RUF015
    assert result == (0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_4() -> None:
    # https://github.com/Quantinuum/pytket-quantinuum/issues/395
    circ = Circuit(1)
    ctrl = circ.add_c_register(name="control", size=1)
    meas = circ.add_c_register(name="measure", size=1)
    circ.add_c_setreg(1, ctrl)
    circ.X(0, condition=ctrl[0])
    circ.add_c_setreg(0, ctrl)
    circ.X(0, condition=ctrl[0])
    circ.add_c_setreg(1, ctrl)
    circ.X(0, condition=ctrl[0])
    circ.Measure(
        qubit=circ.qubits[0],
        bit=meas[0],
    )
    backend = QuantinuumBackend("H2-1LE")
    compiled_circ = backend.get_compiled_circuit(circ, optimisation_level=0)
    result = backend.run_circuit(compiled_circ, n_shots=100, no_opt=True)
    counts = result.get_counts(meas.to_list())
    assert counts == Counter({(0,): 100})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical_5() -> None:
    # https://github.com/Quantinuum/pytket-phir/issues/159
    circ = Circuit()
    targ_reg = circ.add_c_register("targ_reg", 1)
    ctrl_reg = circ.add_c_register("ctrl_reg", 1)
    circ.add_c_not(arg_in=targ_reg[0], arg_out=targ_reg[0], condition=ctrl_reg[0])
    backend = QuantinuumBackend("H2-1LE")
    compiled_circ = backend.get_compiled_circuit(circuit=circ)
    result = backend.run_circuit(circuit=compiled_circ, n_shots=10)
    counts = result.get_counts()
    assert counts == Counter({(0, 0): 10})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_wasm() -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1)
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])

    b = QuantinuumBackend("H2-1LE")

    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, wasm_file_handler=wasfile, n_shots=n_shots).get_counts()
    assert counts == Counter({(1, 0, 0, 0, 0, 0, 0, 0): n_shots})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
# Same as test_wasm_collatz() in backend_test.py but run on local emulators.
def test_wasm_collatz() -> None:
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

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(c, n_shots=10, wasm_file_handler=wasfile)

    r = backend.get_result(h)
    shots = r.get_shots()

    def to_int(C: np.ndarray) -> int:
        assert len(C) == 8
        return int(sum(pow(2, i) * C[i] for i in range(8)))

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


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_midcircuit_measurement_and_reset() -> None:
    c = Circuit(1, 4)
    c.X(0)
    c.Measure(0, 0)
    c.Reset(0)
    c.Measure(0, 1)
    c.X(0)
    c.Measure(0, 2)
    c.Measure(0, 3)

    b = QuantinuumBackend("H2-1LE")

    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots).get_counts()
    assert counts == Counter({(1, 0, 1, 1): n_shots})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_cbits() -> None:
    circ = Circuit(1)
    a = circ.add_c_register("a", 2)
    b = circ.add_c_register("b", 2)

    circ.add_c_setreg(3, a)
    circ.add_c_copyreg(a, b)
    circ.X(0)
    circ.Measure(Qubit(0), a[0])

    backend = QuantinuumBackend("H2-1LE")

    cc = backend.get_compiled_circuit(circ)
    r = backend.run_circuit(cc, n_shots=1)
    counts = r.get_counts(cbits=list(a))
    assert counts == Counter({(1, 1): 1})


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_result_handling_for_empty_bits() -> None:
    # https://github.com/Quantinuum/pytket-quantinuum/issues/473

    circuit = Circuit(1, 2)
    circuit.X(0, condition=circuit.bits[1])

    backend = QuantinuumBackend("H2-1LE")

    compiled_circuit = backend.get_compiled_circuit(circuit=circuit)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=1,
    )
    assert result.get_counts() == {(0, 0): 1}


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_no_noise() -> None:
    # https://github.com/Quantinuum/pytket-quantinuum/issues/571

    backend = QuantinuumBackend("H2-1LE")
    backend_info = backend.backend_info
    assert backend_info is not None
    assert "noise_specs" not in backend_info.misc
