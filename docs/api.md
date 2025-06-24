# API documentation

The pytket-quantinuum extension allows submission of pytket circuits to Quantinuum systems (and emulators) via the {py:class}`QuantinuumBackend`.

See the [pytket-quantinuum section](https://docs.quantinuum.com/systems/trainings/getting_started/pytket_quantinuum/pytket_quantinuum.html) of the documentation website for some example usage.

```{eval-rst}
.. currentmodule:: pytket.extensions.quantinuum
```

```{eval-rst}
.. autoenum:: Language
    :members:
```

```{eval-rst}
.. autoclass:: QuantinuumBackend
    :show-inheritance:
    :special-members: __init__
    :members:
```

```{eval-rst}
.. autoclass:: QuantinuumBackendData
    :members:
.. autodata:: H1
.. autodata:: H2
```

```{eval-rst}
.. autoclass:: QuantinuumBackendCompilationConfig
    :members:
```

```{eval-rst}
.. autoclass:: QuantinuumAPI
    :members:
```

```{eval-rst}
.. autoclass:: QuantinuumAPIOffline
    :members:
```

```{eval-rst}
.. automodule:: pytket.extensions.quantinuum.backends.config
    :members:
```

```{eval-rst}
.. automodule:: pytket.extensions.quantinuum.backends.leakage_gadget
    :members:
```

```{eval-rst}
.. autoclass:: pytket.extensions.quantinuum.backends.credential_storage.CredentialStorage
    :special-members: __init__
    :members:
```

```{eval-rst}
.. autoclass:: pytket.extensions.quantinuum.backends.credential_storage.MemoryCredentialStorage
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: pytket.extensions.quantinuum.backends.credential_storage.QuantinuumConfigCredentialStorage
    :show-inheritance:
```
