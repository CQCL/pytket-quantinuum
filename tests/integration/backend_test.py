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

from base64 import b64encode
from collections import Counter
from pathlib import Path
from typing import cast, Callable, Any  # pylint: disable=unused-import
import json
import gc
import os
import time
from hypothesis import given, settings
import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis.strategies._internal import SearchStrategy
from hypothesis import HealthCheck
from llvmlite.binding import create_context, parse_assembly  # type: ignore
from pytket.backends import CircuitNotValidError
from pytket.predicates import CompilationUnit  # type: ignore

from pytket.circuit import (  # type: ignore
    Circuit,
    Qubit,
    Bit,
    Node,
    OpType,
    reg_eq,
    reg_neq,
    reg_lt,
    reg_gt,
    reg_leq,
    reg_geq,
    if_not_bit,
)
from pytket.extensions.quantinuum import (
    QuantinuumBackend,
    Language,
    prune_shots_detected_as_leaky,
)
from pytket.extensions.quantinuum.backends.quantinuum import (
    GetResultFailed,
    _GATE_SET,
    NoSyntaxChecker,
)
from pytket.extensions.quantinuum.backends.api_wrappers import (
    QuantinuumAPIError,
    QuantinuumAPI,
)
from pytket.backends.status import StatusEnum
from pytket.wasm import WasmFileHandler


skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Quantinuum username)"
)


@pytest.mark.parametrize("authenticated_quum_backend", [None], indirect=True)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_quantinuum(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    if skip_remote_tests:
        backend = QuantinuumBackend(device_name="H1-1SC", machine_debug=True)
    else:
        backend = authenticated_quum_backend
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
    handle = backend.process_circuits([c], n_shots)[0]
    correct_shots = np.zeros((4, 4))
    correct_counts = {(0, 0, 0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    shots = res.get_shots()
    counts = res.get_counts()
    assert backend.circuit_status(handle).status is StatusEnum.COMPLETED
    assert np.all(shots == correct_shots)
    assert counts == correct_counts
    res = backend.run_circuit(c, n_shots=4, timeout=49, language=language)  # type: ignore
    newshots = res.get_shots()
    assert np.all(newshots == correct_shots)
    newcounts = res.get_counts()
    assert newcounts == correct_counts
    if skip_remote_tests:
        assert backend.backend_info is None


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_max_classical_register(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    backend = authenticated_quum_backend

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


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_bell(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots, language=language).get_shots()  # type: ignore
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": "H1-1SC", "label": "test 3"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
    ],
)
@pytest.mark.timeout(120)
def test_multireg(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    gc.disable()
    b = authenticated_quum_backend
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
    shots = b.run_circuit(c, n_shots=n_shots, language=language).get_shots()  # type: ignore
    assert np.array_equal(shots, np.zeros((10, 2)))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_default_pass(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        q0 = Qubit("test0", 5)
        q1 = Qubit("test1", 6)
        c.add_qubit(q0)
        c.H(q0)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c.measure_all()
        c.add_qubit(q1)
        cu = CompilationUnit(c)
        comp_pass.apply(cu)
        # 5 qubits added to Circuit, one is removed when flattening registers
        assert cu.circuit.qubits == [
            Node("q", 0),
            Node("q", 1),
            Node("q", 2),
            Node("q", 3),
        ]
        assert cu.initial_map[Qubit(0)] == Node("q", 0)
        assert cu.initial_map[Qubit(1)] == Node("q", 1)
        assert cu.initial_map[Qubit(2)] == Node("q", 2)
        assert cu.initial_map[q0] == Node("q", 3)
        assert cu.initial_map[q1] == q1
        for pred in b.required_predicates:
            assert pred.verify(cu.circuit)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [
        {"device_name": name, "label": "test cancel"}
        for name in pytest.ALL_SIMULATOR_NAMES  # type: ignore
    ],
    indirect=True,
)
@pytest.mark.timeout(120)
def test_cancel(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    handle = b.process_circuit(c, 10)
    try:
        # will raise HTTP error if job is already completed
        b.cancel(handle)
        time.sleep(1.0)
        assert b.circuit_status(handle).status in [StatusEnum.CANCELLED]
    except QuantinuumAPIError as err:
        check_completed = "job has completed already" in str(err)
        assert check_completed
        if not check_completed:
            raise err


@st.composite
def circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = st.integers(min_value=2, max_value=6),
    depth: SearchStrategy[int] = st.integers(min_value=1, max_value=100),
) -> Circuit:
    total_qubits = draw(n_qubits)
    circuit = Circuit(total_qubits, total_qubits)
    for _ in range(draw(depth)):
        gate = draw(st.sampled_from(list(_GATE_SET)))
        control = draw(st.integers(min_value=0, max_value=total_qubits - 1))
        if gate == OpType.ZZMax:
            target = draw(
                st.integers(min_value=0, max_value=total_qubits - 1).filter(
                    lambda x: x != control
                )
            )
            circuit.add_gate(gate, [control, target])
        elif gate == OpType.Measure:
            circuit.add_gate(gate, [control, control])
            circuit.add_gate(OpType.Reset, [control])
        elif gate == OpType.Rz:
            param = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param], [control])
        elif gate == OpType.PhasedX:
            param1 = draw(st.floats(min_value=0, max_value=2))
            param2 = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param1, param2], [control])
    circuit.measure_all()

    return circuit


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [
        {"device_name": name}
        for name in [
            *pytest.ALL_QUANTUM_HARDWARE_NAMES,  # type: ignore
            *pytest.ALL_SYNTAX_CHECKER_NAMES,  # type: ignore
        ]
    ],
    indirect=True,
)
@given(
    c=circuits(),  # pylint: disable=no-value-for-parameter
    n_shots=st.integers(min_value=1, max_value=10000),
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.timeout(120)
def test_cost_estimate(
    authenticated_quum_backend: QuantinuumBackend,
    c: Circuit,
    n_shots: int,
) -> None:
    b = authenticated_quum_backend
    c = b.get_compiled_circuit(c)
    estimate = None
    if b._device_name.endswith("SC"):
        with pytest.raises(NoSyntaxChecker) as e:
            _ = b.cost(c, n_shots)
        assert "Could not find syntax checker" in str(e.value)
        estimate = b.cost(c, n_shots, syntax_checker=b._device_name, no_opt=False)
    else:
        # All other real hardware backends should have the
        # "syntax_checker" misc property set, so there should be no
        # need of providing it explicitly.
        estimate = b.cost(c, n_shots, no_opt=False)
    if estimate is None:
        pytest.skip("API is flaky, sometimes returns None unexpectedly.")
    assert isinstance(estimate, float)
    assert estimate > 0.0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
    ],
)
@pytest.mark.timeout(120)
def test_classical(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    # circuit to cover capabilities covered in example notebook
    c = Circuit(1, name="test_classical")
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

    if language == Language.QASM:  # remove this when division supported in QIR
        c.add_classicalexpbox_register(a * b // d, d.to_list())

    c.add_classicalexpbox_register(a << 1, a.to_list())
    c.add_classicalexpbox_register(a >> 1, b.to_list())

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.X(0, condition=(a[0] ^ b[0]))
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

    backend = authenticated_quum_backend

    c = backend.get_compiled_circuit(c)
    assert backend.run_circuit(c, n_shots=10, language=language).get_counts()  # type: ignore


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_postprocess(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.add_gate(OpType.PhasedX, [1, 1], [0])
    c.add_gate(OpType.PhasedX, [1, 1], [1])
    c.add_gate(OpType.ZZMax, [0, 1])
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True, language=language)  # type: ignore
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_leakage_detection(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend
    c = Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1)
    h = b.process_circuit(c, n_shots=10, leakage_detection=True)
    r = b.get_result(h)
    assert len(r.c_bits) == 4
    assert sum(r.get_counts().values()) == 10
    r_discarded = prune_shots_detected_as_leaky(r)
    assert len(r_discarded.c_bits) == 2
    assert sum(r_discarded.get_counts().values()) == 10


@given(
    n_shots=st.integers(min_value=1, max_value=10),  # type: ignore
    n_bits=st.integers(min_value=0, max_value=10),  # type: ignore
)
@pytest.mark.timeout(120)
def test_shots_bits_edgecases(n_shots, n_bits) -> None:
    quantinuum_backend = QuantinuumBackend("H1-1SC", machine_debug=True)
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = quantinuum_backend.process_circuit(c, n_shots)
    res = quantinuum_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = quantinuum_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1E"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        # https://github.com/CQCL/pytket-quantinuum/issues/236
        # Language.QIR,
    ],
)
@pytest.mark.timeout(120)
def test_simulator(
    authenticated_quum_handler: QuantinuumAPI,
    authenticated_quum_backend: QuantinuumBackend,
    language: Language,
) -> None:
    circ = Circuit(2, name="sim_test").H(0).CX(0, 1).measure_all()
    n_shots = 1000
    state_backend = authenticated_quum_backend
    stabilizer_backend = QuantinuumBackend(
        "H1-1E", simulator="stabilizer", api_handler=authenticated_quum_handler
    )

    circ = state_backend.get_compiled_circuit(circ)

    noisy_handle = state_backend.process_circuit(circ, n_shots, language=language)  # type: ignore
    pure_handle = state_backend.process_circuit(
        circ, n_shots, noisy_simulation=False, language=language  # type: ignore
    )
    stab_handle = stabilizer_backend.process_circuit(
        circ, n_shots, noisy_simulation=False, language=language  # type: ignore
    )

    noisy_counts = state_backend.get_result(noisy_handle).get_counts()
    assert sum(noisy_counts.values()) == n_shots
    assert len(noisy_counts) > 2  # some noisy results likely

    pure_counts = state_backend.get_result(pure_handle).get_counts()
    assert sum(pure_counts.values()) == n_shots
    assert len(pure_counts) == 2

    stab_counts = stabilizer_backend.get_result(stab_handle).get_counts()
    assert sum(stab_counts.values()) == n_shots
    assert len(stab_counts) == 2

    # test non-clifford circuit fails on stabilizer backend
    # unfortunately the job is accepted, then fails, so have to check get_result
    non_stab_circ = (
        Circuit(2, name="non_stab_circ").H(0).Rx(0.1, 0).CX(0, 1).measure_all()
    )
    non_stab_circ = stabilizer_backend.get_compiled_circuit(non_stab_circ)
    broken_handle = stabilizer_backend.process_circuit(
        non_stab_circ, n_shots, language=language  # type: ignore
    )

    with pytest.raises(GetResultFailed) as _:
        _ = stabilizer_backend.get_result(broken_handle)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_retrieve_available_devices(
    authenticated_quum_backend: QuantinuumBackend,
    authenticated_quum_handler: QuantinuumAPI,
) -> None:
    # authenticated_quum_backend still needs a handler or it will
    # attempt to use the DEFAULT_API_HANDLER.
    backend_infos = authenticated_quum_backend.available_devices(
        api_handler=authenticated_quum_handler
    )
    assert len(backend_infos) > 0
    assert all(
        OpType.ZZPhase in backend_info.gate_set for backend_info in backend_infos
    )


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_batching(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    state_backend = authenticated_quum_backend
    circ = state_backend.get_compiled_circuit(circ)
    # test batch can be resumed

    h1 = state_backend.start_batch(500, circ, 10)
    h2 = state_backend.add_to_batch(h1, circ, 10)
    h3 = state_backend.add_to_batch(h1, circ, 10, batch_end=True)

    assert state_backend.get_results([h1, h2, h3])


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_submission_with_group(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(
        c,
        n_shots=n_shots,
        group=os.getenv("PYTKET_REMOTE_QUANTINUUM_GROUP", default="Default - UK"),
        language=language,  # type: ignore
    ).get_shots()  # type: ignore
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_zzphase(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    backend = authenticated_quum_backend
    c = Circuit(2, 2, "test rzz")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 0)
    c.CY(0, 1)
    c.ZZPhase(0.1, 1, 0)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 0)

    assert c0.n_gates_of_type(backend.default_two_qubit_gate) > 0

    n_shots = 4
    handle = backend.process_circuits([c0], n_shots, language=language)[0]  # type: ignore
    correct_counts = {(0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    counts = res.get_counts()
    assert counts == correct_counts

    c = Circuit(2, 2, "test_rzz_1")
    c.H(0).H(1)
    c.ZZPhase(1, 1, 0)
    c.H(0).H(1)
    c1 = backend.get_compiled_circuit(c, 1)
    assert c1.n_gates_of_type(OpType.ZZPhase) == 0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_zzphase_support_opti2(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    backend = authenticated_quum_backend
    c = Circuit(3, 3, "test rzz synthesis")
    c.H(0)
    c.CX(0, 2)
    c.Rz(0.2, 2)
    c.CX(0, 2)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 2)

    assert c0.n_gates_of_type(backend.default_two_qubit_gate) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_prefer_zzphase(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    # We should prefer small-angle ZZPhase to alternative ZZMax decompositions
    backend = authenticated_quum_backend
    c = (
        Circuit(2)
        .H(0)
        .H(1)
        .ZZPhase(0.1, 0, 1)
        .Rx(0.2, 0)
        .Ry(0.3, 1)
        .ZZPhase(0.1, 0, 1)
        .H(0)
        .H(1)
        .measure_all()
    )
    c0 = backend.get_compiled_circuit(c)
    if backend.default_two_qubit_gate == OpType.ZZPhase:
        assert c0.n_gates_of_type(OpType.ZZPhase) == 2
    elif backend.default_two_qubit_gate == OpType.ZZMax:
        assert c0.n_gates_of_type(OpType.ZZMax) == 2
    else:
        assert backend.default_two_qubit_gate == OpType.TK2
        assert c0.n_gates_of_type(OpType.TK2) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("device_name", pytest.ALL_DEVICE_NAMES)  # type: ignore
@pytest.mark.timeout(120)
def test_device_state(
    device_name: str, authenticated_quum_handler: QuantinuumAPI
) -> None:
    assert isinstance(
        QuantinuumBackend.device_state(
            device_name, api_handler=authenticated_quum_handler
        ),
        str,
    )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        # https://github.com/CQCL/pytket-quantinuum/issues/232
        # Language.QIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent / "testfile.wasm"))
    c = Circuit(1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])

    b = authenticated_quum_backend

    c = b.get_compiled_circuit(c)
    h = b.process_circuits(
        [c], n_shots=10, wasm_file_handler=wasfile, language=language  # type: ignore
    )[0]
    assert b.get_result(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_wasm_costs(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent / "testfile.wasm"))
    c = Circuit(1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])

    b = authenticated_quum_backend

    c = b.get_compiled_circuit(c)
    costs = b.cost(c, n_shots=10, syntax_checker="H1-1SC", wasm_file_handler=wasfile)
    if costs is None:
        pytest.skip("API is flaky, sometimes returns None unexpectedly.")
    assert isinstance(costs, float)
    assert costs > 0.0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_submit_qasm(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    qasm = """
    OPENQASM 2.0;
    include "hqslib1.inc";

    qreg q[2];
    creg c[2];
    U1q(0.5*pi,0.5*pi) q[0];
    measure q[0] -> c[0];
    if(c[0]==1) rz(1.5*pi) q[0];
    if(c[0]==1) rz(0.0*pi) q[1];
    if(c[0]==1) U1q(3.5*pi,0.5*pi) q[1];
    if(c[0]==1) ZZ q[0],q[1];
    if(c[0]==1) rz(3.5*pi) q[1];
    if(c[0]==1) U1q(3.5*pi,1.5*pi) q[1];
    """

    b = authenticated_quum_backend
    h = b.submit_program(Language.QASM, qasm, n_shots=10)
    assert b.get_result(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_options(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    # Unrecognized options are ignored
    c0 = Circuit(1).H(0).measure_all()
    b = authenticated_quum_backend
    c = b.get_compiled_circuit(c0, 0)
    h = b.process_circuits([c], n_shots=1, options={"ignoreme": 0}, language=language)  # type: ignore
    r = b.get_results(h)[0]
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR])
@pytest.mark.timeout(120)
def test_no_opt(
    authenticated_quum_backend: QuantinuumBackend, language: Language
) -> None:
    c0 = Circuit(1).H(0).measure_all()
    b = authenticated_quum_backend
    c = b.get_compiled_circuit(c0, 0)
    h = b.process_circuits([c], n_shots=1, no_opt=True, language=language)  # type: ignore
    r = b.get_results(h)[0]
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_allow_2q_gate_rebase(authenticated_quum_backend: QuantinuumBackend) -> None:
    c0 = Circuit(2).H(0).CX(0, 1).measure_all()
    b = authenticated_quum_backend
    b.set_compilation_config_target_2qb_gate(OpType.ZZMax)
    c = b.get_compiled_circuit(c0, 0)
    h = b.process_circuits([c], n_shots=1, allow_2q_gate_rebase=True)
    r = b.get_results(h)[0]
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 2


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission(authenticated_quum_backend: QuantinuumBackend) -> None:
    # disable Garbage Collector because of
    # https://github.com/CQCL/pytket-quantinuum/issues/170
    gc.disable()
    b = authenticated_quum_backend
    qir = """; ModuleID = 'result_tag.bc'
source_filename = "qat-link"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [5 x i8] c"0_t0\\00"
@1 = internal constant [5 x i8] c"0_t1\\00"

define void @Quantinuum__EntangledState() #0 {
entry:
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__reset__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__result_record_output(%Result* null, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @1, i32 0, i32 0))
  call void @__quantum__rt__tuple_end_record_output()
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare %Result* @__quantum__qis__m__body(%Qubit*)

declare void @__quantum__qis__reset__body(%Qubit*)

declare void @__quantum__rt__qubit_release(%Qubit*)

declare void @__quantum__rt__tuple_start_record_output()

declare void @__quantum__rt__result_record_output(%Result*, i8*)

declare void @__quantum__rt__tuple_end_record_output()

declare void @__quantum__qis__mz__body(%Qubit*, %Result*)

attributes #0 = { "EntryPoint" "maxQubitIndex"="1" "maxResultIndex"="1" "requiredQubits"="2" "requiredResults"="2" }
"""
    ctx = create_context()
    module = parse_assembly(qir, context=ctx)
    ir = module.as_bitcode()
    h = b.submit_program(Language.QIR, b64encode(ir).decode("utf-8"), n_shots=10)
    r = b.get_result(h)
    assert set(r.get_bitlist()) == set([Bit("0_t0", 0), Bit("0_t1", 0)])
    assert len(r.get_shots()) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_conversion(authenticated_quum_backend: QuantinuumBackend) -> None:
    c0 = Circuit(2).H(0).CX(0, 1).measure_all()
    b = authenticated_quum_backend
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10, language=Language.QIR)  # type: ignore
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    assert all(len(shot) == 2 for shot in shots)


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_old_handle(
    authenticated_quum_backend: QuantinuumBackend,
) -> None:
    # https://github.com/CQCL/pytket-quantinuum/issues/189
    c = Circuit(2, 2, "test")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b0 = QuantinuumBackend(
        "H1-1SC",
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            api_url=os.getenv("PYTKET_REMOTE_QUANTINUUM_API_URL"),
            _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
            _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
        ),
    )
    c = b0.get_compiled_circuit(c)
    h0 = b0.process_circuit(c, n_shots=10)
    r0 = b0.get_result(h0)
    b1 = QuantinuumBackend(
        "H1-1SC",
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            api_url=os.getenv("PYTKET_REMOTE_QUANTINUUM_API_URL"),
            _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
            _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
        ),
    )
    h1 = h0[:2]
    r1 = b1.get_result(h1)  # type: ignore
    assert (r0.get_shots() == r1.get_shots()).all()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend", [{"device_name": "H1-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_scratch_removal(authenticated_quum_backend: QuantinuumBackend) -> None:
    # https://github.com/CQCL/pytket-quantinuum/issues/213
    c = Circuit()
    qb0 = c.add_q_register("qb0", 3)
    qb1 = c.add_q_register("qb1", 1)
    cb0 = c.add_c_register("cb0", 2)
    cb1 = c.add_c_register("cb1", 3)

    c.add_gate(OpType.Reset, qb1)  # type:ignore
    c.CX(qb0[0], qb1[0])
    c.CX(qb0[1], qb1[0])
    c.Measure(qb1[0], cb0[0])
    c.add_gate(OpType.Reset, qb1)  # type:ignore
    c.CX(qb0[1], qb1[0])
    c.CX(qb0[2], qb1[0])
    c.Measure(qb1[0], cb0[1])
    c.X(qb0[0], condition=reg_eq(cb0, 1))
    c.X(qb0[2], condition=reg_eq(cb0, 2))
    c.X(qb0[1], condition=reg_eq(cb0, 3))
    c.Measure(qb0[0], cb1[0])
    c.Measure(qb0[1], cb1[1])
    c.Measure(qb0[2], cb1[2])

    b = authenticated_quum_backend
    c1 = b.get_compiled_circuit(c, optimisation_level=1)
    h = b.process_circuit(c1, n_shots=3)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 3
    assert all(len(shot) == 5 for shot in shots)
