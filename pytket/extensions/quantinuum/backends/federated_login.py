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

from typing import Tuple
from http import HTTPStatus
import json
import msal  # type: ignore

AZURE_AD_APP_ID = "4ae73294-a491-45b7-bab4-945c236ee67a"
AZURE_AD_AUTHORITY = "https://login.microsoftonline.com/common"
AZURE_AD_SCOPE = ["User.Read"]


def microsoft_login() -> Tuple[str, str]:
    """Allows a user to login via Microsoft Azure Active Directory"""

    # Create a preferably long-lived app instance which maintains a token cache.
    app = msal.PublicClientApplication(AZURE_AD_APP_ID, authority=AZURE_AD_AUTHORITY)

    # initiate the device flow authorization. It will expire after 15 minutes
    flow = app.initiate_device_flow(scopes=AZURE_AD_SCOPE)

    # check if the device code is available in the flow
    if "user_code" not in flow:
        raise ValueError(
            "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4)
        )

    # this prompts the user to visit https://microsoft.com/devicelogin and
    # enter the provided code on a separate browser/device
    code = flow["user_code"]
    link = flow["verification_uri"]

    print("To sign in:")
    print("1) Open a web browser (using any device)")
    print("2) Visit " + link)
    print("3) Enter code '" + code + "'")
    print("4) Enter your Microsoft credentials")

    # This will block until the we've reached the flow's expiration time
    result = app.acquire_token_by_device_flow(flow)

    # check if we have an ID Token
    if "id_token" in result:
        token = result["id_token"]
        username = result["id_token_claims"]["preferred_username"]
        print("Authentication successful")

    else:

        # Check if a timeout occurred
        if "authorization_pending" in result.get("error"):
            print("Authorization code expired. Please try again.")
        else:
            # some other error occurred
            print(result.get("error"))
            print(result.get("error_description"))
            print(
                result.get("correlation_id")
            )  # You may need this when reporting a bug

        # a token was not returned (an error occurred or the request timed out)
        raise RuntimeError(
            f"Unable to authorize federated login", HTTPStatus.UNAUTHORIZED
        )

    return username, token
