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

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import jwt

from .config import QuantinuumConfig


class CredentialStorage(ABC):
    """Base class for storing Quantinuum username and authentication tokens.

    ``pytket-quantinuum`` interacts with Quantinuum services by making API requests,
    such as submitting quantum programs and retrieving results. These requests
    often require an ID token, which is obtained through the Quantinuum login API.
    The login process also returns a refresh token, allowing the ID token to be
    refreshed without requiring a new login.

    ``CredentialStorage`` defines the interface for storing and accessing these
    credentials, with derived classes providing specific implementations.
    """

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
        """
        :param id_token_timedelt: The time duration for which the ID token is valid.
            Defaults to 55 minutes.
        :param refresh_token_timedelt: The time duration for which the refresh token
            is valid. Defaults to 29 days.
        """
        self._id_timedelt = id_token_timedelt
        self._refresh_timedelt = refresh_token_timedelt

    @abstractmethod
    def save_refresh_token(self, refresh_token: str) -> None:
        """Save refresh token.

        :param refresh_token: refresh token.
        """

    @abstractmethod
    def save_id_token(self, id_token: str) -> None:
        """Save ID token.

        :param id_token: ID token.
        """

    @abstractmethod
    def save_user_name(self, user_name: str) -> None:
        """Save username.

        :param user_name: Quantinuum username.
        """

    def save_tokens(self, id_token: str, refresh_token: str) -> None:
        """Save ID token and refresh token.

        :param id_token: ID token.
        :param refresh_token: refresh token.
        """
        self.save_id_token(id_token)
        self.save_refresh_token(refresh_token)

    @abstractmethod
    def delete_credential(self) -> None:
        """Delete credential."""

    @property  # noqa: B027
    def id_token(self) -> str | None:
        """Return the ID token if valid."""

    @property  # noqa: B027
    def refresh_token(self) -> str | None:
        """Return the refresh token if valid."""

    @property  # noqa: B027
    def user_name(self) -> str | None:
        """Return the username if exists."""


class MemoryCredentialStorage(CredentialStorage):
    """In-memory credential storage.

    This storage option allows credentials to be temporarily stored in memory during
    the application's runtime.
    """

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
        """Construct a MemoryCredentialStorage instance.

        :param id_token_timedelt: The time duration for which the ID token is valid.
            Defaults to 55 minutes.
        :param refresh_token_timedelt: The time duration for which the refresh token
            is valid. Defaults to 29 days.
        """
        super().__init__(id_token_timedelt, refresh_token_timedelt)
        self._user_name: str | None = None
        # Password storage is only included for debug purposes
        self._password: str | None = None
        self._id_token: str | None = None
        self._refresh_token: str | None = None
        self._id_token_timeout: datetime | None = None
        self._refresh_token_timeout: datetime | None = None

    def save_user_name(self, user_name: str) -> None:
        self._user_name = user_name

    def save_refresh_token(self, refresh_token: str) -> None:
        self._refresh_token = refresh_token
        self._refresh_token_timeout = (
            datetime.now(timezone.utc) + self._refresh_timedelt
        )

    def save_id_token(self, id_token: str) -> None:
        self._id_token = id_token
        self._id_token_timeout = datetime.now(timezone.utc) + self._id_timedelt

    @property
    def id_token(self) -> str | None:
        if self._id_token is not None:
            timeout = (
                jwt.decode(
                    self._id_token,
                    algorithms=["HS256"],
                    options={"verify_signature": False},
                )["exp"]
                - 60
            )
            if self._id_token_timeout is not None:
                timeout = min(timeout, self._id_token_timeout.timestamp())
            if datetime.now(timezone.utc).timestamp() > timeout:
                self._id_token = None
        return self._id_token

    @property
    def refresh_token(self) -> str | None:
        if (
            self._refresh_token is not None
            and self._refresh_token_timeout is not None
            and datetime.now(timezone.utc) > self._refresh_token_timeout
        ):
            self._refresh_token = None
        return self._refresh_token

    @property
    def user_name(self) -> str | None:
        return self._user_name

    def delete_credential(self) -> None:
        del self._user_name
        del self._password
        del self._id_token
        del self._refresh_token
        self._user_name = None
        self._password = None
        self._id_token = None
        self._id_token_timeout = None
        self._refresh_token = None
        self._refresh_token_timeout = None


class QuantinuumConfigCredentialStorage(CredentialStorage):
    """Store username and tokens in the default pytket configuration file.

    This storage option allows authentication status to persist beyond the current
    session, reducing the need to re-enter credentials when constructing new
    backends.

    Example:

    >>> backend = QuantinuumBackend(
    >>>     device_name=machine,
    >>>     api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
    >>> )
    """

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
        """Construct a QuantinuumConfigCredentialStorage instance.

        :param id_token_timedelt: The time duration for which the ID token is valid.
            Defaults to 55 minutes.
        :param refresh_token_timedelt: The time duration for which the refresh token
            is valid. Defaults to 29 days.
        """
        super().__init__(id_token_timedelt, refresh_token_timedelt)

    def save_user_name(self, user_name: str) -> None:
        hconfig = QuantinuumConfig.from_default_config_file()
        hconfig.username = user_name
        hconfig.update_default_config_file()

    def save_refresh_token(self, refresh_token: str) -> None:
        hconfig = QuantinuumConfig.from_default_config_file()
        hconfig.refresh_token = refresh_token
        refresh_token_timeout = datetime.now(timezone.utc) + self._refresh_timedelt
        hconfig.refresh_token_timeout = refresh_token_timeout.strftime(
            "%Y-%m-%d %H:%M:%S.%z"
        )
        hconfig.update_default_config_file()

    def save_id_token(self, id_token: str) -> None:
        hconfig = QuantinuumConfig.from_default_config_file()
        hconfig.id_token = id_token
        id_token_timeout = datetime.now(timezone.utc) + self._id_timedelt
        hconfig.id_token_timeout = id_token_timeout.strftime("%Y-%m-%d %H:%M:%S.%z")
        hconfig.update_default_config_file()

    @property
    def id_token(self) -> str | None:
        hconfig = QuantinuumConfig.from_default_config_file()
        id_token = hconfig.id_token
        if id_token is not None:
            timeout = (
                jwt.decode(
                    id_token,
                    algorithms=["HS256"],
                    options={"verify_signature": False},
                )["exp"]
                - 60
            )
            if hconfig.id_token_timeout is not None:
                id_token_timeout = datetime.strptime(
                    hconfig.id_token_timeout, "%Y-%m-%d %H:%M:%S.%z"
                )
                timeout = min(timeout, id_token_timeout.timestamp())
            if datetime.now(timezone.utc).timestamp() > timeout:
                return None
        return id_token

    @property
    def refresh_token(self) -> str | None:
        hconfig = QuantinuumConfig.from_default_config_file()
        refresh_token = hconfig.refresh_token
        if refresh_token is not None and hconfig.refresh_token_timeout is not None:
            refresh_token_timeout = datetime.strptime(
                hconfig.refresh_token_timeout, "%Y-%m-%d %H:%M:%S.%z"
            )
            if datetime.now(timezone.utc) > refresh_token_timeout:
                return None
        return refresh_token

    @property
    def user_name(self) -> str | None:
        hconfig = QuantinuumConfig.from_default_config_file()
        return hconfig.username

    def delete_credential(self) -> None:
        hconfig = QuantinuumConfig.from_default_config_file()
        hconfig.username = None
        hconfig.refresh_token = None
        hconfig.id_token = None
        hconfig.refresh_token_timeout = None
        hconfig.id_token_timeout = None
        hconfig.update_default_config_file()
