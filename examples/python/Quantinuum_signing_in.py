# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Signing into the Quantinuum User Portal

# This notebook contains examples of how to sign in.

# **Requirements:**

# * You must a verified Quantinuum account
# * You must have accepted the latest [terms and conditions](https://um.qapi.honeywell.com/static/media/user_terms_and_conditions.46957d35.pdf) which can only be done by logging into the [Quantinuum User Portal](https://um.qapi.quantinuum.com).

# **Process:**

# There are two ways to login. The option you choose depends on how you set up your account during registration. If you set up your account by registering your email address directly, use Option 1. If you utilized Microsoft credentials to login, use Option 2.

# 1. [Login](#Login)
# 2. [Login via Microsoft](#Login-via-Microsoft)

# ## Login

# If you already have a verified user portal account you will be asked to provide your credentials after initializing the `QuantinuumBackend` interface. If you opted to set up a native account during your user registration we recommend you use this approach.

# * Prompt 1: `Enter your email:`

# * Prompt 2: `Enter your password: `

# If you opted to set up Multi-factor Authentication (MFA), you will also be required to provide a verification `code`. This code automatically refreshes every 30 seconds and can be found on the authenticator app used to setup MFA on your account. The interface does not change when MFA is setup.  To enable MFA navigate to the *Account* tab in the user view on the user portal. MFA is not available for users logging in through a Microsoft account.

# * Prompt 3: `Enter your MFA verification code: `

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"  # Substitute any Quantinuum target
backend = QuantinuumBackend(device_name=machine)
backend.login()

# ## Login via Microsoft

# If you would like to login using a Microsoft account you'll need to set the `provider` flag to `Microsoft` when initializing `QuantinuumBackend`. If you signed up with a Microsoft account during your user registration, you will be required to use this approach.

# Instead of being prompted for your email and password, you will be prompted with this message:

# ```
# To sign in:
# 1) Open a web browser (using any device)
# 2) Visit https://microsoft.com/devicelogin
# 3) Enter code #########
# 4) Enter your Microsoft credentials
# ``````

# As the prompt suggests, you'll need to open this link and enter the provided device `code` to complete your authentication with Microsoft. The Quantinuum API will wait (at most 15 minutes) for these steps to be completed.

# Once authenticated, Quantinuum will verify the federated login request and attempt to issue tokens.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"  # Substitute any Quantinuum target
backend = QuantinuumBackend(device_name=machine, provider="Microsoft")
backend.login()

# **Additional Notes**

# * If Multi-factor Authentication (MFA) is enabled on your account you will also be required to approve this login request.
# * If you don't have access to a web browser you can use another device that does such as a phone or another computer to complete this step.
# * If you receive these error messages:`Unable to complete federated authentication` or `Provider token is invalid`, please check that you are using the same Microsoft account you set up during your registration.

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
