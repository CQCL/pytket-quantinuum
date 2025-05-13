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

"""Quantinuum config."""

from dataclasses import dataclass
from typing import Any, ClassVar

from pytket.config import PytketExtConfig


@dataclass
class QuantinuumConfig(PytketExtConfig):
    """Holds config parameters for pytket-quantinuum."""

    ext_dict_key: ClassVar[str] = "quantinuum"

    username: str | None

    refresh_token: str | None

    id_token: str | None

    refresh_token_timeout: str | None

    id_token_timeout: str | None

    @classmethod
    def from_extension_dict(
        cls: type["QuantinuumConfig"], ext_dict: dict[str, Any]
    ) -> "QuantinuumConfig":
        return cls(
            ext_dict.get("username"),
            ext_dict.get("refresh_token"),
            ext_dict.get("id_token"),
            ext_dict.get("refresh_token_timeout"),
            ext_dict.get("id_token_timeout"),
        )


def set_quantinuum_config(username: str | None) -> None:
    """Set default value for Quantinuum username.
    Can be overriden in backend construction."""
    hconfig = QuantinuumConfig.from_default_config_file()
    hconfig.username = username
    hconfig.update_default_config_file()
