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
"""Methods for generating a leakage detection Pytket Circuit."""


from typing import List, Dict, Tuple, Counter, cast, Sequence
from pytket import Circuit, Qubit, Bit, OpType  # type: ignore
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray

LEAKAGE_DETECTION_BIT_NAME_ = "leakage_detection_bit"
LEAKAGE_DETECTION_QUBIT_NAME_ = "leakage_detection_qubit"


def get_leakage_gadget_circuit(
    circuit_qubit: Qubit, postselection_qubit: Qubit, postselection_bit: Bit
) -> Circuit:
    """
    Returns a two qubit Circuit for detecting leakage errors.

    :param circuit_qubit: Generated circuit detects whether leakage errors
        have occurred in this qubit.
    :type circuit_qubit: Qubit
    :param postselection_qubit: Measured qubit to detect leakage error.
    :type postselection_qubit: Qubit
    :param postselection_bit: Leakage detection result is written to this bit.
    :type postselection_bit: Bit
    :return: Circuit for detecting leakage errors for specified ids.
    :rtype: Circuit
    """
    c = Circuit()
    c.add_qubit(circuit_qubit)
    c.add_qubit(postselection_qubit)
    c.add_bit(postselection_bit)
    c.X(postselection_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.H(postselection_qubit).ZZMax(postselection_qubit, circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.ZZMax(postselection_qubit, circuit_qubit).H(postselection_qubit).Z(circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.Measure(postselection_qubit, postselection_bit)
    c.add_gate(OpType.Reset, [postselection_qubit])
    return c


def get_detection_circuit(circuit: Circuit, n_device_qubits: int) -> Circuit:
    """
    For a passed circuit, appends a leakage detection circuit for
    each end of circuit measurement using spare device qubits.
    All additional Qubit added for leakage detection are
    written to a new register "leakage_detection_qubit" and all
    additional Bit are written to a new register "leakage_detection_bit".

    :param circuit: Circuit to have leakage detection added.
    :type circuit: Circuit
    :param n_device_qubits: Total number of qubits supported by the device
        being compiled to.
    :type n_device_qubits: int

    :return: Circuit with leakage detection circuitry added.
    :rtype: Circuit
    """
    n_qubits: int = circuit.n_qubits
    if n_qubits == 0:
        raise ValueError(
            "Circuit for Leakage Gadget Postselection must have at least one Qubit."
        )
    n_spare_qubits: int = n_device_qubits - n_qubits
    if n_spare_qubits <= 0:
        raise ValueError("Device has no spare qubits for adding leakage detection.")
    # construct detection circuit
    detection_circuit: Circuit = Circuit()
    postselection_qubits: List[Qubit] = [
        Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, i) for i in range(n_spare_qubits)
    ]
    for q in circuit.qubits + postselection_qubits:
        detection_circuit.add_qubit(q)
    for b in circuit.bits:
        detection_circuit.add_bit(b)

    # construct a Circuit that is the original Circuit without
    # end of Circuit Measure gates
    end_circuit_measures: Dict[Qubit, Bit] = {}
    for com in circuit:
        if com.op.type == OpType.Barrier:
            detection_circuit.add_barrier(com.args)
            continue
        # first check if a mid circuit measure needs to be readded
        for q in com.qubits:
            # this condition only true if this Qubit has previously had a
            # "mid-circuit" measure operation
            if q in end_circuit_measures:
                detection_circuit.Measure(q, end_circuit_measures.pop(q))
        if com.op.type == OpType.Measure:
            # if this is "mid-circuit" then this will be rewritten later
            end_circuit_measures[com.qubits[0]] = com.bits[0]
        elif com.op.params:
            detection_circuit.add_gate(com.op.type, com.op.params, com.args)
        else:
            detection_circuit.add_gate(com.op.type, com.args)

    # for each entry in end_circuit_measures, we want to add a leakage_gadget_circuit
    # we try to use each free architecture qubit as few times as possible
    ps_q_index: int = 0
    ps_b_index: int = 0
    for q in end_circuit_measures:
        if q.reg_name == LEAKAGE_DETECTION_QUBIT_NAME_:
            raise ValueError(
                "Leakage Gadget scheme makes a qubit register named "
                "'leakage_detection_qubit' but this already exists in"
                " the passed circuit."
            )
        ps_q_index = 0 if ps_q_index == n_spare_qubits else ps_q_index
        leakage_detection_bit: Bit = Bit(LEAKAGE_DETECTION_BIT_NAME_, ps_b_index)
        if leakage_detection_bit in circuit.bits:
            raise ValueError(
                "Leakage Gadget scheme makes a new Bit named 'leakage_detection_bit'"
                " but this already exists in the passed circuit."
            )
        leakage_gadget_circuit: Circuit = get_leakage_gadget_circuit(
            q, postselection_qubits[ps_q_index], leakage_detection_bit
        )
        detection_circuit.append(leakage_gadget_circuit)
        # increment value for adding postselection to
        ps_q_index += 1
        ps_b_index += 1

    # finally measure the original qubits

    for q, b in end_circuit_measures.items():
        detection_circuit.Measure(q, b)

    detection_circuit.remove_blank_wires()
    return detection_circuit


def prune_shots_detected_as_leaky(result: BackendResult) -> BackendResult:
    """
    For all states with a Bit with name "leakage_detection_bit"
    in a state 1 sets the counts to 0.

    :param result: Shots returned from device.
    :type result: BackendResult
    :return: Shots with leakage cases removed.
    :rtype: BackendResult
    """
    regular_bits: List[Bit] = [
        b for b in result.c_bits if b.reg_name != LEAKAGE_DETECTION_BIT_NAME_
    ]
    leakage_bits: List[Bit] = [
        b for b in result.c_bits if b.reg_name == LEAKAGE_DETECTION_BIT_NAME_
    ]
    received_counts: Counter[Tuple[int, ...]] = result.get_counts(
        cbits=regular_bits + leakage_bits
    )
    discarded_counts: Counter[Tuple[int, ...]] = Counter(
        {
            tuple(state[: len(regular_bits)]): received_counts[state]
            for state in received_counts
            if not any(state[-len(leakage_bits) :])
        }
    )
    return BackendResult(
        counts=Counter(
            {
                OutcomeArray.from_readouts([key]): val
                for key, val in discarded_counts.items()
            }
        ),
        c_bits=cast(Sequence[Bit], regular_bits),
    )
