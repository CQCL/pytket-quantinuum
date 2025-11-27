# API documentation

The pytket-quantinuum extension allows submission of pytket circuits to Quantinuum systems (and emulators) via the {py:class}`~.QuantinuumBackend`.

```{eval-rst}
.. automodule:: pytket.extensions.quantinuum
.. automodule:: pytket.extensions.quantinuum._metadata
.. automodule:: pytket.extensions.quantinuum.backends
.. automodule:: pytket.extensions.quantinuum.backends.quantinuum

    .. autoenum:: Language
        :members:

    .. autoclass:: QuantinuumBackend
        :show-inheritance:
        :special-members: __init__
        :members:

        .. automethod:: pass_from_info


    .. autoclass:: QuantinuumBackendCompilationConfig
        :members:

    .. autoexception:: BackendOfflineError
    .. autoexception:: BatchingUnsupported
    .. autoexception:: DeviceNotAvailable
    .. autoexception:: GetResultFailed
    .. autoexception:: LanguageUnsupported
    .. autoexception:: MaxShotsExceeded
    .. autoexception:: NoSyntaxChecker

.. automodule:: pytket.extensions.quantinuum.backends.data

    .. autoclass:: QuantinuumBackendData
        :members:
    .. autodata:: H2

.. automodule:: pytket.extensions.quantinuum.backends.api_wrappers

    .. autoclass:: QuantinuumAPI

        .. automethod:: cancel
        .. automethod:: delete_authentication
        .. automethod:: full_login
        .. automethod:: get_calendar
        .. automethod:: get_machine_list
        .. automethod:: login
        .. automethod:: retrieve_job
        .. automethod:: retrieve_job_status
        .. automethod:: status

    .. autoclass:: QuantinuumAPIOffline
        :members:

.. automodule:: pytket.extensions.quantinuum.backends.config
    :members:

.. automodule:: pytket.extensions.quantinuum.backends.leakage_gadget
    :members:

.. automodule:: pytket.extensions.quantinuum.backends.credential_storage

    .. autoclass:: CredentialStorage
        :special-members: __init__
        :members:

    .. autoclass:: MemoryCredentialStorage
        :show-inheritance:
        :members:

    .. autoclass:: QuantinuumConfigCredentialStorage
        :show-inheritance:
        :members:

.. autoexception:: pytket.extensions.quantinuum.backends.quantinuum.WasmUnsupported

.. autoexception:: pytket.extensions.quantinuum.backends.api_wrappers.QuantinuumAPIError

.. automodule:: pytket.extensions.quantinuum.backends.federated_login

    .. autofunction:: microsoft_login

.. automodule:: pytket.extensions.quantinuum.backends.calendar_visualisation

    .. autoclass:: QuantinuumCalendar
        :members:
```
