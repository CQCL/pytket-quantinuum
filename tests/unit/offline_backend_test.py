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

import numpy as np
import pytest
from hypothesis import given, strategies

from pytket.backends import CircuitNotValidError
from pytket.circuit import Circuit
from pytket.extensions.quantinuum import Language, QuantinuumBackend
from pytket.extensions.quantinuum.backends.api_wrappers import (
    QuantinuumAPI,
    QuantinuumAPIOffline,
)
from pytket.passes import (
    FullPeepholeOptimise,
    OptimisePhaseGadgets,
    RemoveRedundancies,
    SequencePass,
)


def test_max_classical_register_ii() -> None:
    qapioffline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(
        device_name="H2-1",
        machine_debug=False,
        api_handler=qapioffline,  # type: ignore
    )

    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    assert backend._check_all_circuits([c])  # noqa: SLF001
    for i in range(20):
        c.add_c_register(f"creg-{i}", 32)

    assert backend._check_all_circuits([c])  # noqa: SLF001

    for i in range(20, 5000):
        c.add_c_register(f"creg-{i}", 32)

    with pytest.raises(CircuitNotValidError):
        backend._check_all_circuits([c])  # noqa: SLF001


@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
def test_tket_pass_submission(language: Language) -> None:
    backend = QuantinuumBackend(device_name="H2-1SC", machine_debug=True)

    sequence_pass = SequencePass(
        [
            OptimisePhaseGadgets(),
            FullPeepholeOptimise(),
            FullPeepholeOptimise(allow_swaps=False),
            RemoveRedundancies(),
        ]
    )

    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    n_shots = 4
    backend.process_circuits([c], n_shots, pytketpass=sequence_pass, language=language)


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),
    n_bits=strategies.integers(min_value=0, max_value=10),
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
def test_shots_bits_edgecases(n_shots: int, n_bits: int, language: Language) -> None:
    quantinuum_backend = QuantinuumBackend("H2-1SC", machine_debug=True)
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = quantinuum_backend.process_circuit(c, n_shots, language=language)
    res = quantinuum_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = quantinuum_backend.run_circuit(c, n_shots=n_shots, language=language)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


def test_defaultapi_handler() -> None:
    """Test that the default API handler is used on backend construction."""
    backend_1 = QuantinuumBackend("H2-1")
    backend_2 = QuantinuumBackend("H2-1")

    assert backend_1.api_handler is backend_2.api_handler


def test_custom_api_handler() -> None:
    """Test that custom API handlers are used when used on backend construction."""
    handler_1 = QuantinuumAPI()
    handler_2 = QuantinuumAPI()

    backend_1 = QuantinuumBackend("H2-1", api_handler=handler_1)
    backend_2 = QuantinuumBackend("H2-1", api_handler=handler_2)

    assert backend_1.api_handler is not backend_2.api_handler
