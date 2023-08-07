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

import pytest

from pytket.circuit import Circuit, OpType, Qubit, reg_eq  # type: ignore
from pytket._tket.circuit import _TEMP_BIT_NAME, _TEMP_BIT_REG_BASE  # type: ignore
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.quantinuum.backends.quantinuum import scratch_reg_resize_pass
from pytket.qasm import circuit_to_qasm_str


def test_convert() -> None:
    circ = Circuit(4)
    circ.H(0).CX(0, 1)
    circ.add_gate(OpType.noop, [1])
    circ.CRz(0.5, 1, 2)
    circ.add_barrier([2])
    circ.measure_all()

    QuantinuumBackend("", machine_debug=True).rebase_pass(False).apply(circ)
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

    QuantinuumBackend("", machine_debug=True).rebase_pass(False).apply(circ)
    circ_quum = circuit_to_qasm_str(circ, header="hqslib1")
    qasm_str = circ_quum.split("\n")[6:-1]
    assert all(
        any(com.startswith(gate) for gate in ("rz", "U1q", "ZZ", "measure", "RZZ"))
        for com in qasm_str
    )


def test_resize_scratch_registers() -> None:
    max_c_reg_width_list = [30, 40]
    for max_c_reg_width in max_c_reg_width_list:
        circ = Circuit(1)
        reg_a = circ.add_c_register("a", 1)
        reg_b = circ.add_c_register("b", 1)
        n_scratch_bits = max_c_reg_width * 2 + 2
        for _ in range(n_scratch_bits):
            circ.add_gate(OpType.PhasedX, [1, 0], [0], condition=reg_a[0] ^ reg_b[0])
        original_scratch_reg = circ.get_c_register(_TEMP_BIT_NAME)
        # check the scratch reg size is max_c_reg_width
        assert original_scratch_reg.size == n_scratch_bits

        # apply the resize pass
        c_compiled = circ.copy()
        scratch_reg_resize_pass(max_c_reg_width).apply(c_compiled)

        # check the old register is replaced
        with pytest.raises(RuntimeError) as e:
            c_compiled.get_c_register(_TEMP_BIT_NAME)
        err_msg = "Cannot find classical register"
        assert err_msg in str(e.value)

        # check the new registers have the correct sizes
        scratch_reg1 = c_compiled.get_c_register(f"{_TEMP_BIT_NAME}_0")
        scratch_reg2 = c_compiled.get_c_register(f"{_TEMP_BIT_NAME}_1")
        scratch_reg3 = c_compiled.get_c_register(f"{_TEMP_BIT_NAME}_2")
        assert scratch_reg1.size == max_c_reg_width
        assert scratch_reg2.size == max_c_reg_width
        assert scratch_reg3.size == 2
        args_map = dict()
        original_cmds = circ.get_commands()
        for cmd in original_cmds:
            for arg in cmd.args:
                args_map[arg] = arg

        for i in range(n_scratch_bits):
            args_map[original_scratch_reg[i]] = c_compiled.get_c_register(
                f"{_TEMP_BIT_NAME}_{i//max_c_reg_width}"
            )[i % max_c_reg_width]

        # Check the compiled circuit is equivalent to the original one up to renaming
        compiled_cmds = c_compiled.get_commands()
        for i, cmd in enumerate(original_cmds):
            for j, arg in enumerate(cmd.args):
                assert compiled_cmds[i].args[j] == args_map[arg]

    # If the max width is not exceeded, do nothing
    circ = Circuit(1)
    reg_a = circ.add_c_register("a", 1)
    reg_b = circ.add_c_register("b", 1)
    for _ in range(30):
        circ.add_gate(OpType.PhasedX, [1, 0], [0], condition=reg_a[0] ^ reg_b[0])
    c_compiled = circ.copy()
    scratch_reg_resize_pass(40).apply(c_compiled)
    assert circ == c_compiled

    # Test _TEMP_BIT_REG_BASE is ignored
    circ = Circuit(1, name="test_classical")
    reg_a = circ.add_c_register("a", 1)
    reg_b = circ.add_c_register("b", 1)
    circ.X(0, condition=reg_eq(reg_a ^ reg_b, 1))
    assert circ.get_c_register(f"{_TEMP_BIT_REG_BASE}_0").size == 32
    c_compiled = circ.copy()
    scratch_reg_resize_pass(10).apply(c_compiled)
    assert circ == c_compiled


def test_implicit_swap_removal() -> None:
    b = QuantinuumBackend("", machine_debug=True)
    c = Circuit(2).ISWAPMax(0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 1
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    c = Circuit(2).ISWAPMax(0, 1)
    b.rebase_pass(False).apply(c)
    assert c.n_gates_of_type(OpType.ZZMax) == 2
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).Sycamore(0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    c = Circuit(2).Sycamore(0, 1)
    b.rebase_pass(False).apply(c)
    assert c.n_gates_of_type(OpType.ZZMax) == 3
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ISWAP(0.3, 0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    assert iqp[Qubit(1)] == Qubit(1)
    c = Circuit(2).ISWAP(0.3, 0, 1)
    b.rebase_pass(False).apply(c)
    assert c.n_gates_of_type(OpType.ZZMax) == 2
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 2
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    assert iqp[Qubit(1)] == Qubit(1)
    c = Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0)
    compiled = b.get_compiled_circuit(c, 1)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 0
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(0)
    c = Circuit(2).ISWAPMax(0, 1).ISWAPMax(1, 0)
    b.rebase_pass(False).apply(c)
    assert c.n_gates_of_type(OpType.ZZMax) == 4
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).SWAP(0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates_of_type(OpType.ZZMax) == 0
    assert compiled.n_gates_of_type(OpType.ZZPhase) == 0
    iqp = compiled.implicit_qubit_permutation()
    assert iqp[Qubit(0)] == Qubit(1)
    assert iqp[Qubit(1)] == Qubit(0)
    c = Circuit(2).SWAP(0, 1)
    b.rebase_pass(False).apply(c)
    assert c.n_gates_of_type(OpType.ZZMax) == 3
    assert c.n_gates_of_type(OpType.ZZPhase) == 0

    c = Circuit(2).ZZMax(0, 1)
    compiled = b.get_compiled_circuit(c, 0)
    assert compiled.n_gates == 1


if __name__ == "__main__":
    test_implicit_swap_removal()
