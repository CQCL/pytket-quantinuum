# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Signing into the Quantinuum User Portal

# This notebook contains examples of how to sign in via the Quantinuum API.

# **Requirements:**
# * You must a verified Quantinuum account
# * You must have accepted the latest [terms and conditions](https://um.qapi.honeywell.com/static/media/user_terms_and_conditions.46957d35.pdf) which can only be done by logging into the [Quantinuum User Portal](https://um.qapi.quantinuum.com)

# ## Login <a class="anchor" id="native-login"></a>

# If you already have a verified user portal account you will be asked to provide your credentials after initializing the `QuantinuumBackend` interface. If you opted to set up a native account during your user registration we recommend you use this approach.

# * Prompt 1: `Enter your email:`

# * Prompt 2: `Enter your password: `

# If you opted to set up Multi-factor Authentication (MFA), you will also be required to provide a verification `code`. This code automatically refreshes every 30 seconds and can be found on the authenticator app used to setup MFA on your account. The interface does not change when MFA is setup.  To enable MFA navigate to the *Account* tab in the user view on the user portal. MFA is not available for users logging in through a Microsoft account.

# * Prompt 3: `Enter your MFA verification code: `

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-2E"  # Substitute any Quantinuum target
backend = QuantinuumBackend(device_name=machine)
backend.login()

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
