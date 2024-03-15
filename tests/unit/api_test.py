# Copyright 2020-2024 Cambridge Quantum Computing
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

from io import StringIO
from typing import Any, Tuple
import json

from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)


def test_quum_login(
    mock_quum_api_handler: QuantinuumAPI,
    mock_credentials: Tuple[str, str],
    mock_token: str,
) -> None:
    """Test that credentials are storable and deletable using
    the QuantinuumQAPI handler."""

    _, pwd = mock_credentials

    assert isinstance(mock_quum_api_handler._cred_store, MemoryCredentialStorage)
    # Check credentials are retrievable
    assert mock_quum_api_handler._cred_store._password == pwd
    assert mock_quum_api_handler._cred_store.refresh_token == mock_token
    assert mock_quum_api_handler._cred_store.id_token == mock_token

    # Delete authentication and verify
    mock_quum_api_handler.delete_authentication()
    assert mock_quum_api_handler._cred_store.id_token == None
    assert mock_quum_api_handler._cred_store._password == None
    assert mock_quum_api_handler._cred_store.refresh_token == None


def test_machine_status(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
) -> None:
    """Test that we can retrieve the machine state via Quantinuum endpoint."""

    machine_name = "quum-LT-S1-APIVAL"
    mock_machine_state = "online"

    mock_url = f"https://qapi.quantinuum.com/v1/machine/{machine_name}"

    requests_mock.register_uri(
        "GET",
        mock_url,
        json={"state": mock_machine_state},
        headers={"Content-Type": "application/json"},
    )

    assert mock_quum_api_handler.status(machine_name) == mock_machine_state

    # Delete authentication tokens to clean them from memory
    mock_quum_api_handler.delete_authentication()


def test_full_login(
    requests_mock: Mocker,
    mock_credentials: Tuple[str, str],
    mock_token: str,
    monkeypatch: Any,
) -> None:
    """Test that we can perform the login flow."""
    username, pwd = mock_credentials

    mock_url = "https://qapi.quantinuum.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_token,
            "refresh-token": "refresh" + mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    # fake user input from stdin
    monkeypatch.setattr("sys.stdin", StringIO(username + "\n"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    api_handler = QuantinuumAPI()
    # emulate no pytket config stored email address
    api_handler.full_login()

    assert isinstance(api_handler._cred_store, MemoryCredentialStorage)
    assert api_handler._cred_store.id_token == mock_token
    assert api_handler._cred_store.refresh_token == "refresh" + mock_token
    assert api_handler._cred_store._id_token_timeout is not None
    assert api_handler._cred_store._refresh_token_timeout is not None

    assert api_handler._cred_store._password is None
    assert api_handler._cred_store._user_name is None

    api_handler.delete_authentication()

    assert all(
        val is None
        for val in (
            api_handler._cred_store.id_token,
            api_handler._cred_store.refresh_token,
            api_handler._cred_store._id_token_timeout,
            api_handler._cred_store._refresh_token_timeout,
        )
    )


def test_get_calendar(
    requests_mock: Mocker, mock_quum_api_handler: QuantinuumAPI
) -> None:
    start_date = "2024-02-08"
    end_date = "2024-02-16"

    base = "https://ui.qapi.quantinuum.com/beta/reservation"
    mock_url = f"{base}?mode=user&start={start_date}&end={end_date}"

    events = [
        {
            "start-date": "2024-02-09T00:00:00",
            "machine": "quum-LT-S1-APIVAL",
            "end-date": "2024-02-09T09:00:00",
            "event-type": "online",
            "reservation-type": "",
        },
        {
            "start-date": "2024-02-10T00:00:00",
            "machine": "quum-LT-S2-APIVAL",
            "end-date": "2024-02-10T05:00:00",
            "event-type": "online",
            "reservation-type": "",
        },
    ]

    requests_mock.register_uri(
        "GET",
        mock_url,
        json=json.dumps(events),
        headers={"Content-Type": "application/json"},
    )

    response = json.loads(mock_quum_api_handler.get_calendar(start_date, end_date))
    assert response == events
    mock_quum_api_handler.delete_authentication()
