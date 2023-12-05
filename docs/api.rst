API documentation
~~~~~~~~~~~~~~~~~

The pytket-quantinuum extension allows submission of pytket circuits to the H-series trapped ion devices (and emulators) via the :py:class:`QuantinuumBackend`.

Consult the `Notebooks <https://github.com/CQCL/pytket-quantinuum/tree/develop/examples>`_ for some example usage.

.. currentmodule:: pytket.extensions.quantinuum

.. autoenum:: Language
    :members:

.. autoclass:: QuantinuumBackend
    :show-inheritance:
    :members:

.. autoclass:: QuantinuumBackendCompilationConfig
    :members:

.. autoclass:: QuantinuumAPI
    :members:

.. autoclass:: QuantinuumAPIOffline
    :members:

.. automodule:: pytket.extensions.quantinuum.backends.config
    :members:

.. automodule:: pytket.extensions.quantinuum.backends.leakage_gadget
    :members: