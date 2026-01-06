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

from pytket.architecture import FullyConnected
from pytket.backends.backendinfo import BackendInfo
from pytket.circuit import Circuit, OpType, Qubit
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPIError
from pytket.qasm import circuit_to_qasm_str


def test_convert() -> None:
    circ = Circuit(4)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.noop, [1])
    circ.CRz(0.5, 1, 2)
    circ.add_barrier([2])
    circ.measure_all()

    b = QuantinuumBackend("", machine_debug=True)
    b.set_compilation_config_target_2qb_gate(OpType.ZZMax)
    b.set_compilation_config_allow_implicit_swaps(False)
    b.rebase_pass().apply(circ)
    circ_quum = circuit_to_qasm_str(circ, header="hqslib1")
    qasm_str = circ_quum.split("\n")[6:-1]
    assert all(
        any(com.startswith(gate) for gate in ("rz", "U1q", "ZZ", "measure", "barrier"))
        for com in qasm_str
    )


def test_convert_rzz() -> None:
    circ = Circuit(4)
    circ.Rz(0.5, 1)
    circ.add_gate(OpType.PhasedX, [0.2, 0.3], [1])
    circ.ZZPhase(0.3, 2, 3)
    circ.add_gate(OpType.ZZMax, [2, 3])
    circ.measure_all()

    b = QuantinuumBackend("", machine_debug=True)
    b.set_compilation_config_allow_implicit_swaps(False)
    b.rebase_pass().apply(circ)
    circ_quum = circuit_to_qasm_str(circ, header="hqslib1")
    qasm_str = circ_quum.split("\n")[6:-1]
    assert all(
        any(com.startswith(gate) for gate in ("rz", "U1q", "ZZ", "measure", "RZZ"))
        for com in qasm_str
    )


def test_implicit_swap_removal() -> None:  # noqa: PLR0915
    b = QuantinuumBackend("", machine_debug=True)
    c = Circuit(2).ISWAPMax(0, 1)
    b.set_compilation_config_target_2qb_gate(OpType.ZZMax)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 1
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    b.set_compilation_config_allow_implicit_swaps(False)
    c = b.get_compiled_circuit(Circuit(2).ISWAPMax(0, 1), 0)
    assert c.n_gates_of_type(OpType.ZZMax) == 2
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).Sycamore(0, 1)
    b.set_compilation_config_allow_implicit_swaps(True)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    b.set_compilation_config_allow_implicit_swaps(False)
    c = b.get_compiled_circuit(Circuit(2).Sycamore(0, 1), 0)
    assert c.n_gates_of_type(OpType.ZZMax) == 3
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ISWAP(0.3, 0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    assert iqp[Qubit(1)] == Qubit(1)
    c = b.get_compiled_circuit(Circuit(2).ISWAP(0.3, 0, 1), 0)
    assert c.n_gates_of_type(OpType.ZZMax) == 2
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0)
    b.set_compilation_config_allow_implicit_swaps(True)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    assert iqp[Qubit(1)] == Qubit(1)
    c = Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0)
    compiled = b.get_compiled_circuit(c, 1)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    b.set_compilation_config_allow_implicit_swaps(False)
    c = b.get_compiled_circuit(Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0), 0)
    assert c.n_gates_of_type(OpType.ZZMax) == 4
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).SWAP(0, 1)
    b.set_compilation_config_allow_implicit_swaps(True)
    b.set_compilation_config_target_2qb_gate(OpType.ZZPhase)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 0
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    b.set_compilation_config_allow_implicit_swaps(False)
    b.set_compilation_config_target_2qb_gate(OpType.ZZMax)
    c = b.get_compiled_circuit(Circuit(2).SWAP(0, 1), 0)
    assert c.n_gates_of_type(OpType.ZZMax) == 3
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ZZMax(0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates == 1


def test_switch_target_2qb_gate() -> None:
    # this default device only "supports" ZZMax
    b = QuantinuumBackend("", machine_debug=True)
    c = Circuit(2).ISWAPMax(0, 1)
    # Default behaviour
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 0
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 1
    assert compiled.n_gates_of_type(OpType.TK2) == 0
    # Targeting allowed gate
    b.set_compilation_config_target_2qb_gate(OpType.ZZMax)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 1
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    assert compiled.n_gates_of_type(OpType.TK2) == 0
    # Targeting allowed gate but no wire swap
    b.set_compilation_config_allow_implicit_swaps(False)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    assert compiled.n_gates_of_type(OpType.TK2) == 0
    # Targeting unsupported gate
    with pytest.raises(QuantinuumAPIError):
        b.set_compilation_config_target_2qb_gate(OpType.ISWAPMax)

    # Confirming that if ZZPhase is added to gate set that it functions
    b._MACHINE_DEBUG = False  # noqa: SLF001
    b._backend_info = BackendInfo(  # noqa: SLF001
        name="test",
        device_name="test",
        version="test",
        architecture=FullyConnected(1),
        gate_set={OpType.ZZPhase, OpType.ZZMax, OpType.PhasedX, OpType.Rz},
    )
    assert OpType.ZZMax in b._gate_set  # noqa: SLF001
    assert OpType.ZZPhase in b._gate_set  # noqa: SLF001
    b.set_compilation_config_allow_implicit_swaps(True)
    b.set_compilation_config_target_2qb_gate(OpType.ZZPhase)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 0
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 1
    assert compiled.n_gates_of_type(OpType.TK2) == 0


def test_not_allow_implicit_swaps() -> None:
    # https://github.com/Quantinuum/pytket-quantinuum/issues/515
    b = QuantinuumBackend("", machine_debug=True)
    circuits = [Circuit(2).SWAP(0, 1), Circuit(2).CX(0, 1).CX(1, 0).CX(0, 1)]
    b.set_compilation_config_allow_implicit_swaps(False)
    for target_gate in [OpType.ZZMax, OpType.ZZPhase, OpType.TK2]:
        for c in circuits:
            b.set_compilation_config_target_2qb_gate(target_gate)
            d0 = b.get_compiled_circuit(c, optimisation_level=0)
            d1 = b.get_compiled_circuit(c, optimisation_level=1)
            d2 = b.get_compiled_circuit(c, optimisation_level=2)
            assert d0.implicit_qubit_permutation() == {
                Qubit(0): Qubit(0),
                Qubit(1): Qubit(1),
            }
            assert d1.implicit_qubit_permutation() == {
                Qubit(0): Qubit(0),
                Qubit(1): Qubit(1),
            }
            assert d2.implicit_qubit_permutation() == {
                Qubit(0): Qubit(0),
                Qubit(1): Qubit(1),
            }


if __name__ == "__main__":
    test_implicit_swap_removal()
    test_switch_target_2qb_gate()
    test_not_allow_implicit_swaps()
