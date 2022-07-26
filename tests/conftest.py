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

import os
from typing import Any, Dict, List, Tuple

import pytest
from _pytest.fixtures import SubRequest
from requests_mock.mocker import Mocker
import jwt

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None


@pytest.fixture()
def mock_credentials() -> Tuple[str, str]:
    username = "mark.quantinuum@mail.com"
    pwd = "1906"
    return (username, pwd)


@pytest.fixture()
def mock_token() -> str:
    # A mock token that expires in 2073
    token_payload = {"exp": 3278815149.143694}
    mock_token = jwt.encode(token_payload, key="", algorithm="HS256")
    return mock_token


@pytest.fixture()
def mock_mfa_code() -> str:
    return "mfa code"


@pytest.fixture()
def mock_ms_provider_token() -> str:
    return "ms token"


@pytest.fixture()
def mock_machine_info() -> Dict[str, Any]:
    return {
        "name": "H9-27",
        "n_qubits": 12,
        "gateset": [],
        "n_classical_registers": 50,
        "n_shots": 10000,
        "system_family": "mock_family",
        "system_type": "hardware",
        "emulator": "H9-27E",
        "syntax_checker": "H9-27SC",
        "batching": True,
        "wasm": True,
    }


@pytest.fixture()
def sample_machine_infos() -> List[Dict[str, Any]]:
    return [
        {
            "name": "H1-2SC",
            "n_qubits": 12,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_type": "syntax checker",
            "wasm": True,
        },
        {
            "name": "H1-1SC",
            "n_qubits": 12,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_type": "syntax checker",
            "wasm": True,
        },
        {
            "name": "H1-1E",
            "n_qubits": 20,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_type": "emulator",
            "batching": True,
            "wasm": True,
        },
        {
            "name": "H1-1",
            "n_qubits": 20,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_family": "H1",
            "system_type": "hardware",
            "emulator": "H1-1E",
            "syntax_checker": "H1-1SC",
            "batching": True,
            "wasm": True,
        },
        {
            "name": "H1-2",
            "n_qubits": 12,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_family": "H1",
            "system_type": "hardware",
            "emulator": "H1-2E",
            "syntax_checker": "H1-2SC",
            "batching": True,
            "wasm": True,
        },
        {
            "name": "H1-2E",
            "n_qubits": 12,
            "gateset": ["RZZ", "Riswap", "Rxxyyzz"],
            "n_classical_registers": 50,
            "n_shots": 10000,
            "system_type": "emulator",
            "batching": True,
            "wasm": True,
        },
        {"name": "H1", "n_qubits": 20},
    ]


@pytest.fixture(name="mock_quum_api_handler", params=[True, False])
def fixture_mock_quum_api_handler(
    request: SubRequest,
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
) -> QuantinuumAPI:
    """A logged-in QuantinuumQAPI fixture.
    After using this fixture in a test, call:
        mock_quum_api_handler.delete_authentication()
    To remove mock tokens from the keyring.
    """

    username, pwd = mock_credentials

    mock_url = "https://qapi.quantinuum.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    cred_store = MemoryCredentialStorage()
    cred_store._save_login_credential(
        user_name=username,
        password=pwd,
    )

    # Construct QuantinuumQAPI and login
    api_handler = QuantinuumAPI()

    # Add the credential storage seperately in line with fixture parameters
    api_handler.config.username = username
    api_handler._cred_store = cred_store
    api_handler.login()

    return api_handler


@pytest.fixture(scope="module", name="authenticated_quum_handler")
def fixture_authenticated_quum() -> QuantinuumAPI:
    # Authenticated QuantinuumAPI used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME and PYTKET_REMOTE_QUANTINUUM_PASSWORD
    return QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
        _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
        _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
    )


@pytest.fixture(name="authenticated_quum_backend")
def fixture_authenticated_quum_backend(
    request: SubRequest,
    authenticated_quum_handler: QuantinuumAPI,
) -> QuantinuumBackend:
    # Authenticated QuantinuumBackend used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME and PYTKET_REMOTE_QUANTINUUM_PASSWORD
    # Note: this fixture should only be used in tests where PYTKET_RUN_REMOTE_TESTS
    #       is true, by marking it with @parametrize, using the
    #       "authenticated_quum_backend" as parameter and `indirect=True`

    # By default, the backend is created with device_name="H1-1SC" only,
    # but other params can be specified when parametrizing the
    # authenticated_quum_backend
    if request.param is None:
        backend = QuantinuumBackend("H1-1SC", _api_handler=authenticated_quum_handler)
    else:
        backend = QuantinuumBackend(
            _api_handler=authenticated_quum_handler, **request.param
        )
    # In case machine_debug was specified by mistake in the params
    backend._MACHINE_DEBUG = False

    return backend
