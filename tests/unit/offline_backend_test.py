# Copyright 2020-2024 Quantinuum
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


@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
def test_quantinuum_offline(language: Language) -> None:
    qapioffline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(
        device_name="H1-1", machine_debug=False, api_handler=qapioffline  # type: ignore
    )
    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    n_shots = 4
    _ = backend.process_circuits([c], n_shots, language=language)[0]
    expected_result = {
        "name": "test 1",
        "count": 4,
        "machine": "H1-1",
        "language": "OPENQASM 2.0" if language == Language.QASM else "QIR 1.0",
        "program": "...",  # not checked
        "priority": "normal",
        "options": {"simulator": "state-vector", "error-model": True, "tket": {}},
    }
    result = qapioffline.get_jobs()
    assert result is not None
    assert result[0]["name"] == expected_result["name"]
    assert result[0]["count"] == expected_result["count"]
    assert result[0]["machine"] == expected_result["machine"]
    assert result[0]["language"] == expected_result["language"]
    assert result[0]["priority"] == expected_result["priority"]
    # assert result[0]["options"] == expected_result["options"]


def test_max_classical_register_ii() -> None:
    qapioffline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(
        device_name="H1-1", machine_debug=False, api_handler=qapioffline  # type: ignore
    )

    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    assert backend._check_all_circuits([c])
    for i in range(0, 20):
        c.add_c_register(f"creg-{i}", 32)

    assert backend._check_all_circuits([c])

    for i in range(20, 200):
        c.add_c_register(f"creg-{i}", 32)

    with pytest.raises(CircuitNotValidError):
        backend._check_all_circuits([c])


@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
def test_tket_pass_submission(language: Language) -> None:
    backend = QuantinuumBackend(device_name="H1-1SC", machine_debug=True)

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
    n_shots=strategies.integers(min_value=1, max_value=10),  # type: ignore
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
def test_shots_bits_edgecases(n_shots, n_bits, language: Language) -> None:
    quantinuum_backend = QuantinuumBackend("H1-1SC", machine_debug=True)
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


@pytest.mark.parametrize("device_name", pytest.ALL_DEVICE_NAMES)  # type: ignore
def test_defaultapi_handler(device_name: str) -> None:
    """Test that the default API handler is used on backend construction."""
    backend_1 = QuantinuumBackend(device_name, machine_debug=True)
    backend_2 = QuantinuumBackend(device_name, machine_debug=True)

    assert backend_1.api_handler is backend_2.api_handler


@pytest.mark.parametrize("device_name", pytest.ALL_DEVICE_NAMES)  # type: ignore
def test_custom_api_handler(device_name: str) -> None:
    """Test that custom API handlers are used when used on backend construction."""
    handler_1 = QuantinuumAPI()
    handler_2 = QuantinuumAPI()

    backend_1 = QuantinuumBackend(
        device_name, api_handler=handler_1, machine_debug=True
    )
    backend_2 = QuantinuumBackend(
        device_name, api_handler=handler_2, machine_debug=True
    )

    assert backend_1.api_handler is not backend_2.api_handler
    assert backend_1.api_handler._cred_store is not backend_2.api_handler._cred_store
