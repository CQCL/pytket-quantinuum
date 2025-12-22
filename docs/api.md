# API documentation

The pytket-quantinuum extension allows submission of pytket circuits to Quantinuum systems (and emulators) via the {py:class}`~.QuantinuumBackend`.

See the [pytket-quantinuum section](https://docs.quantinuum.com/systems/trainings/h2/getting_started/index.html) of the documentation website for some example usage.

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

        .. automethod:: get_machine_list

    .. autoclass:: QuantinuumAPIOffline
        :members:

.. automodule:: pytket.extensions.quantinuum.backends.config
    :members:

.. autoexception:: pytket.extensions.quantinuum.backends.quantinuum.WasmUnsupported

.. autoexception:: pytket.extensions.quantinuum.backends.api_wrappers.QuantinuumAPIError
```
