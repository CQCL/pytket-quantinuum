# Copyright 2020-2022 Cambridge Quantum Computing
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

# NB: This test has been placed in a separate file from api_test.py to work around an
# issue on the MacOS CI, whereby pytest would hang indefinitely after the collection
# phase.

from io import StringIO
from typing import Any, Dict, List, Tuple
from http import HTTPStatus
from unittest.mock import patch, MagicMock
import pytest

import requests
from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends import QuantinuumBackend
from pytket.circuit import Circuit  # type: ignore
from pytket.architecture import FullyConnected  # type: ignore
from pytket.extensions.quantinuum.backends.quantinuum import (
    DEFAULT_API_HANDLER,
    BatchingUnsupported,
)
from pytket.extensions.quantinuum._metadata import __extension_version__


def test_default_login_flow(
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
    mock_machine_info: Dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Test that when an api_handler is not provided to
    QuantinuumBackend we use the DEFAULT_API_HANDLER.

    Demonstrate that the login endpoint is only called one time
    for the session when not providing an api_handler argument to
    QuantinuumBackend.
    """

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]
    fake_job_id = "abc-123"

    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    job_submit_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    job_status_route = requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    username, pwd = mock_credentials
    # fake user input from stdin
    monkeypatch.setattr("sys.stdin", StringIO(username + "\n"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    backend_1 = QuantinuumBackend(
        device_name=fake_device,
    )
    backend_2 = QuantinuumBackend(
        device_name=fake_device,
    )

    circ = Circuit(2, name="default_login_flow_test").H(0).CX(0, 1).measure_all()
    circ = backend_1.get_compiled_circuit(circ)
    circ = backend_2.get_compiled_circuit(circ)

    backend_1.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    backend_2.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )

    # We expect /login to be called once globally.
    assert login_route.called_once  # type: ignore
    assert job_submit_route.call_count == 4  # type: ignore
    assert job_status_route.call_count == 0  # type: ignore


def test_custom_login_flow(
    requests_mock: Mocker,
    mock_token: str,
    mock_machine_info: Dict[str, Any],
) -> None:
    """Test that when an api_handler is provided to
    QuantinuumBackend we use that handler and acquire
    tokens for each.
    """

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]
    fake_job_id = "abc-123"

    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    job_submit_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    job_status_route = requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    backend_1 = QuantinuumBackend(
        device_name=fake_device,
        _api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            _QuantinuumAPI__user_name="user1",
            _QuantinuumAPI__pwd="securepassword",
        ),
    )
    backend_2 = QuantinuumBackend(
        device_name=fake_device,
        _api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            _QuantinuumAPI__user_name="user2",
            _QuantinuumAPI__pwd="insecurepassword",
        ),
    )

    circ = Circuit(2, name="default_login_flow_test").H(0).CX(0, 1).measure_all()
    circ = backend_1.get_compiled_circuit(circ)
    circ = backend_2.get_compiled_circuit(circ)

    backend_1.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    backend_2.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )

    # We expect /login to be called for each api_handler.
    assert login_route.call_count == 2  # type: ignore
    assert job_submit_route.call_count == 4  # type: ignore
    assert job_status_route.call_count == 0  # type: ignore


def test_mfa_login_flow(
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
    mock_mfa_code: str,
    mock_machine_info: Dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Test that the MFA authentication works as expected"""

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    def match_mfa_request(request: requests.PreparedRequest) -> bool:
        return "code" in request.body  # type: ignore

    def match_normal_request(request: requests.PreparedRequest) -> bool:
        return "code" not in request.body  # type: ignore

    mfa_login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
        additional_matcher=match_mfa_request,  # type: ignore
    )
    normal_login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "error": {"code": 73},
        },
        headers={"Content-Type": "application/json"},
        additional_matcher=match_normal_request,  # type: ignore
        status_code=HTTPStatus.UNAUTHORIZED,
    )

    username, pwd = mock_credentials
    # fake user input from stdin
    inputs = iter([username + "\n", mock_mfa_code + "\n"])
    monkeypatch.setattr("builtins.input", lambda msg: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    backend = QuantinuumBackend(
        device_name=fake_device,
    )
    backend.login()

    assert normal_login_route.called_once  # type: ignore
    # Check that the mfa login has been invoked
    assert mfa_login_route.called_once  # type: ignore
    assert backend._api_handler._cred_store.id_token is not None
    assert backend._api_handler._cred_store.refresh_token is not None


@patch("pytket.extensions.quantinuum.backends.api_wrappers.microsoft_login")
def test_federated_login(
    mock_microsoft_login: MagicMock,
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
    mock_ms_provider_token: str,
    mock_machine_info: Dict[str, Any],
) -> None:
    """Test that the federated authentication works as expected"""
    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    backend = QuantinuumBackend(
        device_name=fake_device,
        provider="microsoft",
    )
    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )
    mock_microsoft_login.return_value = (mock_credentials[0], mock_ms_provider_token)
    backend.login()

    mock_microsoft_login.assert_called_once()
    assert login_route.called_once  # type: ignore
    assert backend._api_handler._cred_store.id_token is not None
    assert backend._api_handler._cred_store.refresh_token is not None


def test_federated_login_wrong_provider(
    mock_machine_info: Dict[str, Any],
) -> None:
    """Test that the federated authentication works as expected"""
    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    backend = QuantinuumBackend(
        device_name=fake_device,
        provider="wrong provider",
    )
    with pytest.raises(RuntimeError) as e:
        backend.login()
        err_msg = "Unsupported provider for login"
        assert err_msg in str(e.value)


@pytest.mark.parametrize(
    "chosen_device",
    ["H1", "H1-1", "H1-2"],
)
def test_device_family(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    sample_machine_infos: List[Dict[str, Any]],
    chosen_device: str,
) -> None:
    """Test that batch params are NOT supplied by default
    if we are submitting to a device family.
    Doing so will get an error response from the Quantinuum API."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/machine/?config=true",
        json=sample_machine_infos,
        headers={"Content-Type": "application/json"},
    )

    backend = QuantinuumBackend(
        device_name=chosen_device,
    )
    backend._api_handler = mock_quum_api_handler

    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    circ = backend.get_compiled_circuit(circ)

    max_batch_cost = 20
    if chosen_device == "H1":
        with pytest.raises(BatchingUnsupported):
            backend.start_batch(max_batch_cost, circ, 10)
    else:
        backend.start_batch(max_batch_cost, circ, 10)
        submitted_json = {}
        if requests_mock.last_request:
            # start_batch makes two requests
            submitted_json = requests_mock.request_history[-2].json()
        assert "batch-exec" in submitted_json
        assert submitted_json["batch-exec"] == max_batch_cost


def test_resumed_batching(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    sample_machine_infos: Dict[str, Any],
) -> None:
    """Test that you can resume using a batch."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/machine/?config=true",
        json=sample_machine_infos,
        headers={"Content-Type": "application/json"},
    )

    backend = QuantinuumBackend(
        device_name="H1-1E",
    )
    backend._api_handler = mock_quum_api_handler

    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    circ = backend.get_compiled_circuit(circ)

    h1 = backend.start_batch(500, circ, n_shots=10, valid_check=False)

    submitted_json = {}
    if requests_mock.last_request:
        # start batch makes two requests
        submitted_json = requests_mock.request_history[-2].json()

    assert "batch-exec" in submitted_json
    assert submitted_json["batch-exec"] == 500
    assert "batch-end" not in submitted_json

    _ = backend.add_to_batch(h1, circ, n_shots=10, valid_check=False, batch_end=True)

    if requests_mock.last_request:
        submitted_json = requests_mock.last_request.json()
    assert submitted_json["batch-exec"] == backend.get_jobid(h1)
    assert "batch-end" in submitted_json


def test_available_devices(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    mock_machine_info: Dict[str, Any],
) -> None:

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    devices = QuantinuumBackend.available_devices(_api_handler=mock_quum_api_handler)
    assert len(devices) == 1
    backinfo = devices[0]

    assert backinfo.device_name == mock_machine_info["name"]
    assert backinfo.architecture == FullyConnected(mock_machine_info["n_qubits"])
    assert backinfo.version == __extension_version__
    assert backinfo.supports_fast_feedforward == True
    assert backinfo.supports_midcircuit_measurement == True
    assert backinfo.supports_reset == True
    assert backinfo.misc == {
        "n_classical_registers": 50,
        "n_shots": 10000,
        "system_family": "mock_family",
        "system_type": "hardware",
        "emulator": "H9-27E",
        "syntax_checker": "H9-27SC",
        "batching": True,
        "wasm": True,
    }
    assert backinfo.name == "QuantinuumBackend"
