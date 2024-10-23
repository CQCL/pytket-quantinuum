# API documentation

The pytket-quantinuum extension allows submission of pytket circuits to the H-series trapped ion devices (and emulators) via the {py:class}`QuantinuumBackend`.

Consult the [notebooks](https://github.com/CQCL/pytket-quantinuum/tree/develop/examples) for some example usage.

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
