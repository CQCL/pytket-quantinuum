import pytest

from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.predicates import CliffordCircuitPredicate


@pytest.mark.parametrize(
    "simulator_type,should_have_clifford_predicate",
    [
        ("state-vector", False),
        ("stabilizer", True),
    ],
)
def test_clifford_circuits_for_stabilizer(
    simulator_type: str,
    should_have_clifford_predicate: bool,
) -> None:
    """Check that the stabilizer simulator restricts circuits to be Clifford circuits,
    and the statevector simulator does not."""
    qapi_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(
        "H2-1E",
        api_handler=qapi_offline,  # type: ignore
        simulator=simulator_type,
    )
    required_predicates = backend.required_predicates
    assert should_have_clifford_predicate == any(
        isinstance(pred, CliffordCircuitPredicate) for pred in required_predicates
    )
