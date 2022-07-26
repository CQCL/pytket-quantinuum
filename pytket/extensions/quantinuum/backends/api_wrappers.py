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

"""
Functions used to submit jobs with Quantinuum API.
"""

import time
from http import HTTPStatus
from typing import Optional, Dict, Tuple
import asyncio
import json
import getpass
import requests
from requests.models import Response
from websockets import connect, exceptions  # type: ignore
import nest_asyncio  # type: ignore

from .config import QuantinuumConfig
from .credential_storage import MemoryCredentialStorage

# This is necessary for use in Jupyter notebooks to allow for nested asyncio loops
try:
    nest_asyncio.apply()
except (RuntimeError, ValueError):
    # May fail in some cloud environments: ignore.
    pass


class QuantinuumAPIError(Exception):
    pass


class _OverrideManager:
    def __init__(
        self,
        api_handler: "QuantinuumAPI",
        timeout: Optional[int] = None,
        retry_timeout: Optional[int] = None,
    ):
        self._timeout = timeout
        self._retry = retry_timeout
        self.api_handler = api_handler
        self._orig_timeout = api_handler.timeout
        self._orig_retry = api_handler.retry_timeout

    def __enter__(self) -> None:
        if self._timeout is not None:
            self.api_handler.timeout = self._timeout
        if self._retry is not None:
            self.api_handler.retry_timeout = self._retry

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.api_handler.timeout = self._orig_timeout
        self.api_handler.retry_timeout = self._orig_retry


class QuantinuumAPI:

    JOB_DONE = ["failed", "completed", "canceled"]

    DEFAULT_API_URL = "https://qapi.quantinuum.com/"

    def __init__(
        self,
        token_store: Optional[MemoryCredentialStorage] = None,
        api_url: Optional[str] = None,
        api_version: int = 1,
        use_websocket: bool = True,
        __user_name: Optional[str] = None,
        __pwd: Optional[str] = None,
    ):
        """Initialize Quantinuum API client.

        :param token_store: JWT Token store, defaults to None
            A new MemoryCredentialStorage will be initialised
            if None is provided.
        :type token_store: MemoryCredentialStorage, optional
        :param api_url: _description_, defaults to DEFAULT_API_URL
        :type api_url: Optional[str], optional
        :param api_version: API version, defaults to 1
        :type api_version: int, optional
        :param use_websocket: Whether to use websocket to retrieve, defaults to True
        :type use_websocket: bool, optional
        """
        self.config = QuantinuumConfig.from_default_config_file()

        self.url = f"{api_url if api_url else self.DEFAULT_API_URL}v{api_version}/"

        if token_store is None:
            self._cred_store = MemoryCredentialStorage()
        else:
            self._cred_store = token_store

        if __user_name is not None:
            self.config.username = __user_name
        if self.config.username is not None and __pwd is not None:
            self._cred_store._save_login_credential(self.config.username, __pwd)

        self.api_version = api_version
        self.use_websocket = use_websocket

        self.ws_timeout = 180
        self.retry_timeout = 5
        self.timeout: Optional[int] = None  # don't timeout by default

    def override_timeouts(
        self, timeout: Optional[int] = None, retry_timeout: Optional[int] = None
    ) -> _OverrideManager:
        return _OverrideManager(self, timeout=timeout, retry_timeout=retry_timeout)

    def _request_tokens(self, user: str, pwd: str) -> None:
        """Method to send login request to machine api and save tokens."""
        body = {"email": user, "password": pwd}
        try:
            # send request to login
            response = requests.post(
                f"{self.url}login",
                json.dumps(body),
            )
            self._response_check(response, "Login")
            resp_dict = response.json()
            self._cred_store.save_tokens(
                resp_dict["id-token"], resp_dict["refresh-token"]
            )

        finally:
            del user
            del pwd
            del body

    def _refresh_id_token(self, refresh_token: str) -> None:
        """Method to refresh ID token using a refresh token."""
        body = {"refresh-token": refresh_token}
        try:
            # send request to login
            response = requests.post(
                f"{self.url}login",
                json.dumps(body),
            )

            message = response.json()

            if (
                response.status_code == HTTPStatus.BAD_REQUEST
                and message is not None
                and "Invalid Refresh Token" in message["error"]["text"]
            ):
                # ask user for credentials to login again
                self.full_login()

            else:
                self._response_check(response, "Token Refresh")
                self._cred_store.save_tokens(
                    message["id-token"], message["refresh-token"]
                )

        finally:
            del refresh_token
            del body

    def _get_credentials(self) -> Tuple[str, str]:
        """Method to ask for user's credentials"""
        user_name = self._cred_store._user_name or self.config.username
        if not user_name:
            user_name = input("Enter your Quantinuum email: ")
        pwd = self._cred_store._password

        if not pwd:
            pwd = getpass.getpass(prompt="Enter your Quantinuum password: ")

        return user_name, pwd

    def full_login(self) -> None:
        """Ask for user credentials from std input and update JWT tokens"""
        self._request_tokens(*self._get_credentials())

    def login(self) -> str:
        """This methods checks if we have a valid (non-expired) id-token
        and returns it, otherwise it gets a new one with refresh-token.
        If refresh-token doesn't exist, it asks user for credentials.
        """
        # check if refresh_token exists
        refresh_token = self._cred_store.refresh_token
        if refresh_token is None:
            self.full_login()
            refresh_token = self._cred_store.refresh_token

        if refresh_token is None:
            raise QuantinuumAPIError(
                "Unable to retrieve refresh token or authenticate."
            )

        # check if id_token exists
        id_token = self._cred_store.id_token
        if id_token is None:
            self._refresh_id_token(refresh_token)
            id_token = self._cred_store.id_token

        if id_token is None:
            raise QuantinuumAPIError("Unable to retrieve id token or refresh or login.")

        return id_token

    def delete_authentication(self) -> None:
        """Remove stored credentials and tokens"""
        self._cred_store._delete_login_credential()
        self._cred_store.delete_tokens()

    def _submit_job(self, body: Dict) -> Response:
        id_token = self.login()
        # send job request
        return requests.post(
            f"{self.url}job",
            json.dumps(body),
            headers={"Authorization": id_token},
        )

    def _response_check(self, res: requests.Response, description: str) -> None:
        """Consolidate as much error-checking of response"""
        # check if token has expired or is generally unauthorized
        if res.status_code == HTTPStatus.UNAUTHORIZED:
            jr = res.json()
            raise QuantinuumAPIError(
                (
                    f"Authorization failure attempting: {description}."
                    f"\n\nServer Response: {jr}"
                )
            )
        elif res.status_code != HTTPStatus.OK:
            jr = res.json()
            raise QuantinuumAPIError(
                f"HTTP error attempting: {description}.\n\nServer Response: {jr}"
            )

    def retrieve_job_status(
        self, job_id: str, use_websocket: Optional[bool] = None
    ) -> Optional[Dict]:
        """
        Retrieves job status from device.

        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction

        Returns:
            (dict):        output from API

        """
        job_url = f"{self.url}job/{job_id}"
        # Using the login wrapper we will automatically try to refresh token
        id_token = self.login()
        if use_websocket or (use_websocket is None and self.use_websocket):
            job_url += "?websocket=true"
        res = requests.get(job_url, headers={"Authorization": id_token})

        jr: Optional[Dict] = None
        # Check for invalid responses, and raise an exception if so
        self._response_check(res, "job status")
        # if we successfully got status return the decoded details
        if res.status_code == HTTPStatus.OK:
            jr = res.json()
        return jr

    def retrieve_job(
        self, job_id: str, use_websocket: Optional[bool] = None
    ) -> Optional[Dict]:
        """
        Retrieves job from device.

        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction

        Returns:
            (dict):        output from API

        """
        jr = self.retrieve_job_status(job_id, use_websocket)
        if not jr:
            raise QuantinuumAPIError(f"Unable to retrive job {job_id}")
        if "status" in jr and jr["status"] in self.JOB_DONE:
            return jr

        if "websocket" in jr:
            # wait for job completion using websocket
            jr = asyncio.get_event_loop().run_until_complete(self._wait_results(job_id))
        else:
            # poll for job completion
            jr = self._poll_results(job_id)
        return jr

    def _poll_results(self, job_id: str) -> Optional[Dict]:
        jr = None
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            try:
                jr = self.retrieve_job_status(job_id)

                # If we are failing to retrieve status of any kind, then fail out.
                if jr is None:
                    break
                if "status" in jr and jr["status"] in self.JOB_DONE:
                    return jr
                time.sleep(self.retry_timeout)
            except KeyboardInterrupt:
                raise RuntimeError("Keyboard Interrupted")
        return jr

    async def _wait_results(self, job_id: str) -> Optional[Dict]:
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            jr = self.retrieve_job_status(job_id, True)
            if jr is None:
                return jr
            elif "status" in jr and jr["status"] in self.JOB_DONE:
                return jr
            else:
                task_token = jr["websocket"]["task_token"]
                execution_arn = jr["websocket"]["executionArn"]
                websocket_uri = self.url.replace("https://", "wss://ws.")
                async with connect(websocket_uri) as websocket:
                    body = {
                        "action": "OpenConnection",
                        "task_token": task_token,
                        "executionArn": execution_arn,
                    }
                    await websocket.send(json.dumps(body))
                    while True:
                        try:
                            res = await asyncio.wait_for(
                                websocket.recv(), timeout=self.ws_timeout
                            )
                            jr = json.loads(res)
                            if not isinstance(jr, Dict):
                                raise RuntimeError("Unable to decode response.")
                            if "status" in jr and jr["status"] in self.JOB_DONE:
                                return jr
                        except (
                            asyncio.TimeoutError,
                            exceptions.ConnectionClosed,
                        ):
                            try:
                                # Try to keep the connection alive...
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=10)
                                continue
                            except asyncio.TimeoutError:
                                # If we are failing, wait a little while,
                                #  then start from the top
                                await asyncio.sleep(self.retry_timeout)
                                break
                        except KeyboardInterrupt:
                            raise RuntimeError("Keyboard Interrupted")

    def status(self, machine: str) -> str:
        """
        Check status of machine.

        Args:
            (str):    machine name

        """
        id_token = self.login()
        res = requests.get(
            f"{self.url}machine/{machine}",
            headers={"Authorization": id_token},
        )
        self._response_check(res, "get machine status")
        jr = res.json()

        return str(jr["state"])

    def cancel(self, job_id: str) -> dict:
        """
        Cancels job.

        Args:
            job_id:     job ID to cancel

        Returns:
            jr:         (dict) output from API

        """

        id_token = self.login()
        res = requests.post(
            f"{self.url}job/{job_id}/cancel", headers={"Authorization": id_token}
        )
        self._response_check(res, "job cancel")
        jr = res.json()

        return jr  # type: ignore
