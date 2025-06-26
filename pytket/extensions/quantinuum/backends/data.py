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

from dataclasses import dataclass

from pytket.circuit import OpType


@dataclass
class QuantinuumBackendData:
    """Data characterizing a QuantinuumBackend.

    * `n_qubits`: maximum number of qubits available
    * `n_cl_reg`: maximum number of classical registers available
    * `gate_set`: set of available native gates
    * `local_emulator`: whether the backend is a local emulator
    """

    n_qubits: int
    n_cl_reg: int
    gate_set: frozenset[OpType] = frozenset(
        {
            OpType.Rz,
            OpType.PhasedX,
            OpType.ZZPhase,
            OpType.Measure,
            OpType.Reset,
        }
    )
    local_emulator: bool = False


"""Data characterizing H1 devices and emulators"""
H1 = QuantinuumBackendData(
    n_qubits=20,
    n_cl_reg=4000,
    gate_set=frozenset(
        {OpType.PhasedX, OpType.Rz, OpType.TK2, OpType.ZZMax, OpType.ZZPhase}
    ),
)


"""Data characterizing H2 devices and emulators"""
H2 = QuantinuumBackendData(
    n_qubits=56,
    n_cl_reg=4000,
    gate_set=frozenset(
        {OpType.PhasedX, OpType.Rz, OpType.TK2, OpType.ZZMax, OpType.ZZPhase}
    ),
)
