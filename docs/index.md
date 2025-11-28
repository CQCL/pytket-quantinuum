---
file_format: mystnb
---

# pytket-quantinuum

`pytket-quantinuum` is an extension to `pytket` that allows `pytket` circuits to
be compiled for execution on Quantinuum's quantum devices. As of version 0.56.0
it no longer allows submission to Quantinuum devices: please use the `qnexus`
package for that.

`pytket-quantinuum` is available for Python 3.10, 3.11 and 3.12, on Linux, MacOS
and Windows. To install, run:

```
pip install pytket-quantinuum
```

# Available Backends

The pytket-quantinuum extension allows the user to access the following quantum emulators. These backends can be initialised  by passing the device name as a string to the {py:class}`~.QuantinuumBackend` class. The available devices are:

- `H2-1LE`, a version of the `H2-1E` emulator that runs locally. For running simulations locally, see the docs on [Local Emulators](#local-emulators).

# Default Compilation

Every {py:class}`~pytket.backends.backend.Backend` in pytket has its own {py:meth}`~pytket.backends.backend.Backend.default_compilation_pass` method.
This method applies a sequence of optimisations to a circuit depending on the value of an `optimisation_level` parameter.
This default compilation will ensure that the circuit meets all the constraints required to run on the {py:class}`~pytket.backends.backend.Backend`.

The default pass can be applied in place as follows

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend

circ = Circuit(2).H(0).CX(0, 1).CZ(0, 1)
backend = QuantinuumBackend('H2-1E')

# Compile the circuit in place. The optimisation level is set to 2 by default.
backend.default_compilation_pass().apply(circ)
```

Alternatively the default pass can be applied using the {py:meth}`~pytket.backends.backend.Backend.get_compiled_circuit` method.

```{code-cell} ipython3
---
tags: [skip-execution]
---
compiled_circ = backend.get_compiled_circuit(circ)
```

The passes applied by different levels of optimisation are specified in the table below. Note that optimisation level 0, 1 and 2 do not remove barriers from 
a circuit, while optimisation level 3 will. At optimisation level 3 the default timeout is 5 minutes - consider increasing this for larger circuits if 
the circuit 2-qubit gate count is not reduced after compilation.

:::{list-table} **Default compilation pass for the QuantinuumBackend**
:widths: 25 25 25 25
:header-rows: 1

* - optimisation_level = 0
  - optimisation_level = 1
  - optimisation_level = 2 [1]
  - optimisation_level = 3
* - {py:meth}`~pytket.passes.DecomposeBoxes`
  - {py:meth}`~pytket.passes.DecomposeBoxes`
  - {py:meth}`~pytket.passes.DecomposeBoxes`
  - {py:meth}`~pytket.passes.DecomposeBoxes`
* - {py:func}`~pytket.passes.resizeregpass.scratch_reg_resize_pass`
  - {py:func}`~pytket.passes.resizeregpass.scratch_reg_resize_pass`
  - {py:func}`~pytket.passes.resizeregpass.scratch_reg_resize_pass`
  - {py:func}`~pytket.passes.resizeregpass.scratch_reg_resize_pass`
* - {py:meth}`~pytket.passes.AutoRebase` [2]
  - {py:meth}`~pytket.passes.SynthesiseTket`
  - {py:meth}`~pytket.passes.FullPeepholeOptimise` [3]
  - {py:meth}`~pytket.passes.RemoveBarriers`
* - {py:meth}`~pytket.passes.FlattenRelabelRegistersPass`
  - {py:meth}`~pytket.passes.NormaliseTK2` [5]
  - {py:meth}`~pytket.passes.NormaliseTK2` [5]
  - {py:meth}`~pytket.passes.GreedyPauliSimp`
* -
  - {py:meth}`~pytket.passes.DecomposeTK2` [5]
  - {py:meth}`~pytket.passes.DecomposeTK2` [5]
  - {py:meth}`~pytket.passes.NormaliseTK2` [5]
* -
  - {py:meth}`~pytket.passes.AutoRebase` [2]
  - {py:meth}`~pytket.passes.AutoRebase` [2]
  - {py:meth}`~pytket.passes.DecomposeTK2` [5]
* -
  - {py:meth}`~pytket.passes.ZZPhaseToRz`
  - {py:meth}`~pytket.passes.RemoveRedundancies`
  - {py:meth}`~pytket.passes.AutoRebase` [2]
* -
  - {py:meth}`~pytket.passes.RemoveRedundancies`
  - {py:meth}`~pytket.passes.AutoSquash` [4]
  - {py:meth}`~pytket.passes.RemoveRedundancies`
* -
  - {py:meth}`~pytket.passes.AutoSquash` [4]
  - {py:meth}`~pytket.passes.FlattenRelabelRegistersPass`
  - {py:meth}`~pytket.passes.AutoSquash` [4]
* -
  - {py:meth}`~pytket.passes.FlattenRelabelRegistersPass`
  - {py:meth}`~pytket.passes.RemoveRedundancies`
  - {py:meth}`~pytket.passes.FlattenRelabelRegistersPass`
* -
  - {py:meth}`~pytket.passes.RemoveRedundancies`
  -
  - {py:meth}`~pytket.passes.RemoveRedundancies`
:::

- \[1\] If no value is specified then `optimisation_level` defaults to a value of 2.
- \[2\] {py:meth}`~pytket.passes.AutoRebase` is a rebase that converts the circuit to the Quantinuum native gate set (e.g. $\{Rz, PhasedX, ZZMax, ZZPhase\}$).
- \[3\] {py:meth}`~pytket.passes.FullPeepholeOptimise` has the argument `target_2qb_gate=OpType.TK2`.
- \[4\] {py:meth}`~pytket.passes.AutoSquash` targets the $\{PhasedX, Rz\}$ gate set, i.e. `AutoSquash({OpType.PhasedX, OpType.Rz})`.
- \[5\] Omitted if the target two-qubit gate is `OpType.TK2`.

:::{note}
If `optimisation_level = 0` the device constraints are solved but no additional optimisation is applied. Setting `optimisation_level = 1` applies some light optimisations to the circuit. More intensive optimisation is applied by level 2 at the expense of increased runtime.
:::

:::{note}
The pass {py:meth}`~pytket.passes.ZZPhaseToRz` is left out of `optimisation_level=2` as the passes applied by {py:meth}`~pytket.passes.FullPeepholeOptimise` will already cover these optimisations.
:::

# Target Two-Qubit Gate

Backends may offer several alternatives as the native two-qubit gate: the
current possibilities are `ZZMax`, `ZZPhase` and `TK2`. The set of
supported gates may be queried using the
{py:attr}`~.QuantinuumBackend.two_qubit_gate_set` property. Each device also has a
default two-qubit gate, which may be queried using the
{py:attr}`~.QuantinuumBackend.default_two_qubit_gate` property. Currently, the default
two-qubit gate for all devices is `ZZPhase`.

The default compilation pass and rebase pass will target the default gate by
default. This may be overridden using the method
{py:meth}`~.QuantinuumBackend.set_compilation_config_target_2qb_gate` or by passing a
{py:class}`~.QuantinuumBackendCompilationConfig` when constructing the backend.

# Device Predicates

Circuits must satisfy the following predicates in order to run on the {py:class}`~.QuantinuumBackend`.

- {py:class}`~pytket.predicates.NoSymbolsPredicate`: Parameterised gates must have numerical parameters when the circuit is executed.
- {py:class}`~pytket.predicates.GateSetPredicate`: To view supported Ops run `QuantinuumBackend.backend_info.gate_set`.
- {py:class}`~pytket.predicates.MaxNQubitsPredicate`: `H2-1`, `H2-1E` and `H2-1SC` all support a maximum of 56 qubits.

# Local Emulators

If `pytket-quantinuum` is installed with the `pecos` option:

```
pip install pytket-quantinuum[pecos]
```

For `uv` virtual enviroments it is possible that this does not work, because prereleases are not picked up automatically. This can be solved by installing the latest `pytket-pecos` version via:

```
uv pip install pytket-pecos --prerelease=allow
```

then it is possible to run circuits on an emulator running on the local machine.

Currently this emulation is noiseless.

```{eval-rst}
.. toctree::
    api.md
    changelog.md
```

```{eval-rst}
.. toctree::
   :caption: Useful links

   Issue tracker <https://github.com/CQCL/pytket-quantinuum/issues>
   PyPi <https://pypi.org/project/pytket-quantinuum/>
```
