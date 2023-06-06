# Copyright 2019-2023 Cambridge Quantum Computing
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


from pytket import Circuit, Qubit, Bit, OpType
from pytket.backends.backendresult import BackendResult  # type: ignore
from pytket.extensions.quantinuum.backends.leakage_gadget import (
    get_detection_circuit,
    prune_shots_detected_as_leaky,
)
from pytket.utils.outcomearray import OutcomeArray  # type: ignore
from typing import Counter, Sequence, cast


def test_postselection_circuits_1qb_task_gen() -> None:
    comparison_circuit: Circuit = Circuit(1, 1)
    lg_qb = Qubit("leakage_detection_qubit", 0)
    lg_b = Bit("leakage_detection_bit", 0)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b)
    comparison_circuit.X(lg_qb).H(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb).ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0)).H(lg_qb).Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0).Measure(lg_qb, lg_b)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])
    assert comparison_circuit == get_detection_circuit(
        Circuit(1, 1).H(0).Measure(0, 0), 2
    )


def test_postselection_circuits_2qb_2_spare_task_gen() -> None:
    comparison_circuit = Circuit(2, 2)
    lg_qb0 = Qubit("leakage_detection_qubit", 0)
    lg_b0 = Bit("leakage_detection_bit", 0)
    lg_qb1 = Qubit("leakage_detection_qubit", 1)
    lg_b1 = Bit("leakage_detection_bit", 1)
    comparison_circuit.add_qubit(lg_qb0)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_qubit(lg_qb1)
    comparison_circuit.add_bit(lg_b1)
    comparison_circuit.X(lg_qb0).X(lg_qb1).H(0).CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.H(lg_qb0).H(lg_qb1).ZZMax(lg_qb0, Qubit(0)).ZZMax(
        lg_qb1, Qubit(1)
    )
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.ZZMax(lg_qb0, Qubit(0)).ZZMax(lg_qb1, Qubit(1))
    comparison_circuit.H(lg_qb0).H(lg_qb1).Z(0).Z(1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.Measure(0, 0).Measure(1, 1).Measure(lg_qb0, lg_b0).Measure(
        lg_qb1, lg_b1
    )
    comparison_circuit.add_gate(OpType.Reset, [lg_qb0])
    comparison_circuit.add_gate(OpType.Reset, [lg_qb1])

    assert comparison_circuit == get_detection_circuit(
        Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1), 4
    )


def test_postselection_circuits_2qb_1_spare_task_gen() -> None:
    comparison_circuit = Circuit(2, 2)
    lg_qb = Qubit("leakage_detection_qubit", 0)
    lg_b0 = Bit("leakage_detection_bit", 0)
    lg_b1 = Bit("leakage_detection_bit", 1)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_bit(lg_b1)

    comparison_circuit.X(lg_qb).H(0).CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb).ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0)).H(lg_qb).Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0).Measure(lg_qb, lg_b0)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])
    comparison_circuit.X(lg_qb)
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.H(lg_qb).ZZMax(lg_qb, Qubit(1))
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(1)).H(lg_qb).Z(1)
    comparison_circuit.add_barrier([Qubit(1), lg_qb])
    comparison_circuit.Measure(1, 1).Measure(lg_qb, lg_b1)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])

    assert comparison_circuit == get_detection_circuit(
        Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1), 3
    )


def test_postselection_discard_0() -> None:
    counter_dict = {(0, 0): 100, (0, 1): 75, (1, 0): 50, (1, 1): 25}
    counts = Counter(
        {OutcomeArray.from_readouts([key]): val for key, val in counter_dict.items()}
    )

    discard_result = prune_shots_detected_as_leaky(
        BackendResult(
            counts=counts,
            c_bits=cast(Sequence[Bit], [Bit(0), Bit("leakage_detection_bit", 0)]),
        )
    ).get_counts()
    assert discard_result[(0,)] == 100
    assert discard_result[(1,)] == 50


def test_postselection_discard_1() -> None:
    raw_readouts = (
        [[0, 0, 0, 0]] * 100
        + [[0, 1, 0, 0]] * 75
        + [[0, 1, 0, 1]] * 50
        + [[1, 0, 1, 0]] * 33
    )
    outcomes = OutcomeArray.from_readouts(raw_readouts)

    backres_shots = BackendResult(
        shots=outcomes,
        c_bits=[
            Bit(3),
            Bit("leakage_detection_bit", 23),
            Bit("a", 7),
            Bit("leakage_detection_bit", 3),
        ],
    )
    discard_result = prune_shots_detected_as_leaky(backres_shots).get_counts()
    assert discard_result[(0, 0)] == 100
    assert discard_result[(1, 1)] == 33


if __name__ == "__main__":
    test_postselection_circuits_1qb_task_gen()
    test_postselection_circuits_2qb_2_spare_task_gen()
    test_postselection_circuits_2qb_1_spare_task_gen()
    test_postselection_discard_0()
    test_postselection_discard_1()
