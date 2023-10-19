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

from abc import ABC, abstractmethod
from typing import Optional
from datetime import timedelta, datetime, timezone
import jwt
from .config import QuantinuumConfig


class CredentialStorage(ABC):

    """Storage for Quantinuum username and tokens"""

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
        self._id_timedelt = id_token_timedelt
        self._refresh_timedelt = refresh_token_timedelt

    @abstractmethod
    def save_refresh_token(self, refresh_token: str) -> None:
        """save refresh token"""

    @abstractmethod
    def save_id_token(self, id_token: str) -> None:
        """save ID token"""

    @abstractmethod
    def save_user_name(self, user_name: str) -> None:
        """save user_name"""

    def save_tokens(self, id_token: str, refresh_token: str) -> None:
        self.save_id_token(id_token)
        self.save_refresh_token(refresh_token)

    @abstractmethod
    def delete_credential(self) -> None:
        """delete credential"""

    @property
    def id_token(self) -> Optional[str]:
        """returns a ID token if valid"""

    @property
    def refresh_token(self) -> Optional[str]:
        """returns a refresh token if valid"""

    @property
    def user_name(self) -> Optional[str]:
        """returns the user name"""


class MemoryCredentialStorage(CredentialStorage):

    """In memory credential storage. Intended use is only to store id tokens,
    refresh tokens and user_name. Password storage is only included for debug
    purposes."""

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
        super().__init__(id_token_timedelt, refresh_token_timedelt)
        self._user_name: Optional[str] = None
        self._password: Optional[str] = None
        self._id_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._id_token_timeout: Optional[datetime] = None
        self._refresh_token_timeout: Optional[datetime] = None

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
    def id_token(self) -> Optional[str]:
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
    def refresh_token(self) -> Optional[str]:
        if self._refresh_token is not None and self._refresh_token_timeout is not None:
            if datetime.now(timezone.utc) > self._refresh_token_timeout:
                self._refresh_token = None
        return self._refresh_token

    @property
    def user_name(self) -> Optional[str]:
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

    """Store tokens in the default pytket configuration file."""

    def __init__(
        self,
        id_token_timedelt: timedelta = timedelta(minutes=55),
        refresh_token_timedelt: timedelta = timedelta(days=29),
    ) -> None:
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
    def id_token(self) -> Optional[str]:
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
    def refresh_token(self) -> Optional[str]:
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
    def user_name(self) -> Optional[str]:
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
