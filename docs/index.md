---
file_format: mystnb
---

# pytket-quantinuum

`pytket-quantinuum` is an extension to `pytket` that allows `pytket` circuits to
be executed on Quantinuum's quantum devices.

`pytket-quantinuum` is available for Python 3.10, 3.11 and 3.12, on Linux, MacOS
and Windows. To install, run:

```
pip install pytket-quantinuum
```

:::{note}
pytket-quantinuum is not compatible with Quantinuum Nexus. For guidance on how to access H-Series through Nexus, please see the [Nexus documentation](https://docs.quantinuum.com/nexus) and the [qnexus](https://pypi.org/project/qnexus/) Python package.
:::

:::{note}
Running circuits remotely on the `QuantinuumBackend` requires a [Quantinuum](https://www.quantinuum.com/) account.
The user will be prompted for their login credentials when making API calls such as calling the `process_circuits` method or querying `backend_info`.
:::

**User Interface:**
<https://um.qapi.quantinuum.com/>

`pytket-quantinuum` provides calendar visualization capabilities. The `calendar` extra-install-argument must be specified to install matplotlib and pandas.

```
pip install pytket-quantinuum[calendar]
```

:::{note}
`view_calendar` requires the extra-install-argument above. The method `get_calendar` can be used with the base installation of pytket-quantinuum.
:::

In case of questions about the hardware you can get in contact with the team sending an email to <mailto:QCsupport@quantinuum.com>.

# Available Backends

The pytket-quantinuum extension allows the user to access the following quantum devices, emulators and syntax checkers. These backends can be initialised  by passing the device name as a string to the `QuantinuumBackend` class. The available devices are:

- `H1-1`, `H2-1`: Quantum devices, submit specifically to `H1-1` or `H2-1` by using the device name.
- `H1-1E`, `H2-1E`: Device-specific emulators of `H1-1` and `H2-1`. These emulators run remotely on servers and require credentials.
- `H1-1SC`, `H2-1SC`: Device-specific syntax checkers. These check compilation of a quantum circuit against device-specific instructions, and return status "completed" if the syntax is correct (along with the H-Series Quantum Credits (HQCs)), or status "failed" if the syntax is incorrect (along with the error).
- `H1-1LE`, a version of the `H1-1E` emulator that runs locally. For running simulations locally see the docs on [Local Emulators].

There are also optional initialisation parameters `label` (for naming circuits), `group` (identifier for a collection of jobs) and `simulator` (see below).

The H-series devices and emulators produce shots-based results and therefore require measurements. It is also possible to use a stabilizer simulator by specifying `simulator='stabilizer'`. This option may be preferable for simulating Clifford circuits.

By default the emulators use noise models based on the real devices. It is possible to perform a noiseless simulation by specifying `noisy_simulation=False`.

For examples demonstrating the `QuantinuumBackend` see the [example notebooks](https://github.com/CQCL/pytket-quantinuum/tree/main/examples) .

# Default Compilation

Every `Backend` in pytket has its own `default_compilation_pass` method.
This method applies a sequence of optimisations to a circuit depending on the value of an `optimisation_level` parameter.
This default compilation will ensure that the circuit meets all the constraints required to run on the `Backend`.

The default pass can be applied in place as follows

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend

circ = Circuit(2).H(0).CX(0, 1).CZ(0, 1)
backend = QuantinuumBackend('H1-1E')

# Compile the circuit in place. The optimisation level is set to 2 by default.
backend.default_compilation_pass().apply(circ)
```

Alternatively the default pass can be applied using the `get_compiled_circuit` method.

```
compiled_circ = backend.get_compiled_circuit(circ)
```

The passes applied by different levels of optimisation are specified in the table below.

:::{list-table} **Default compilation pass for the QuantinuumBackend**
:widths: 25 25 25
:header-rows: 1

* - optimisation_level = 0
  - optimisation_level = 1
  - optimisation_level = 2 [1]
* - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
  - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
  - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
* - [AutoRebase [2]](inv:#*.AutoRebase)
  - [SynthesiseTket](inv:#*.SynthesiseTket)
  - [FullPeepholeOptimise [3]](inv:#*.passes.FullPeepholeOptimise)
* - [FlattenRelabelRegistersPass](inv:#*.FlattenRelabelRegistersPass)
  - [NormaliseTK2 [5]](inv:#*.passes.NormaliseTK2)
  - [NormaliseTK2 [5]](inv:#*.passes.NormaliseTK2)
* -
  - [DecomposeTK2 [5]](inv:#*.passes.DecomposeTK2)
  - [DecomposeTK2 [5]](inv:#*.passes.DecomposeTK2)
* -
  - [AutoRebase [2]](inv:#*.AutoRebase)
  - [AutoRebase [2]](inv:#*.AutoRebase)
* -
  - [ZZPhaseToRz](inv:#*.passes.ZZPhaseToRz)
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
* -
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
  - [AutoSquash [4]](inv:#*.AutoSquash)
* -
  - [AutoSquash [4]](inv:#*.AutoSquash)
  - [FlattenRelabelRegistersPass](inv:#*.FlattenRelabelRegistersPass)
* -
  - [FlattenRelabelRegistersPass](inv:#*.FlattenRelabelRegistersPass)
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
* -
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
  -
:::

- \[1\] If no value is specified then `optimisation_level` defaults to a value of 2.
- \[2\] [AutoRebase](inv:#*.AutoRebase) is a rebase that converts the circuit to the Quantinuum native gate set (e.g. $\{Rz, PhasedX, ZZMax, ZZPhase\}$).
- \[3\] [FullPeepholeOptimise](inv:#*.passes.FullPeepholeOptimise) has the argument `target_2qb_gate=OpType.TK2`.
- \[4\] [AutoSquash](inv:#*.AutoSquash) targets the $\{PhasedX, Rz\}$ gate set, i.e. [AutoSquash({OpType.PhasedX, OpType.Rz}](inv:#*.AutoSquash)`.
- \[5\] Omitted if the target two-qubit gate is `OpType.TK2`.

:::{note}
If `optimisation_level = 0` the device constraints are solved but no additional optimisation is applied. Setting `optimisation_level = 1` applies some light optimisations to the circuit. More intensive optimisation is applied by level 2 at the expense of increased runtime.
:::

:::{note}
The pass [ZZPhaseToRz](inv:#*.passes.ZZPhaseToRz) is left out of `optimisation_level=2` as the passes applied by [FullPeepholeOptimise](inv:#*.passes.FullPeepholeOptimise) will already cover these optimisations.
:::

# Target Two-Qubit Gate

Backends may offer several alternatives as the native two-qubit gate: the
current possibilities are `ZZMax`, `ZZPhase` and `TK2`. The set of
supported gates may be queried using the
`QuantinuumBackend.two_qubit_gate_set` property. Each device also has a
default two-qubit gate, which may be queried using the
`QuantinuumBackend.default_two_qubit_gate` property. Currently, the default
two-qubit gate for all devices is `ZZPhase`.

The default compilation pass and rebase pass will target the default gate by
default. This may be overridden using the method
`QuantinuumBackend.set_compilation_config_target_2qb_gate()` or by passing a
`QuantinuumBackendCompilationConfig` when constructing the backend.

# Device Predicates

Circuits must satisfy the following predicates in order to run on the `QuantinuumBackend`.

- [NoSymbolsPredicate](inv:#*.predicates.NoSymbolsPredicate): Parameterised gates must have numerical parameters when the circuit is executed.
- [GateSetPredicate](inv:#*.predicates.GateSetPredicate): To view supported Ops run `QuantinuumBackend.backend_info.gate_set`.
- [MaxNQubitsPredicate](inv:#*.predicates.MaxNQubitsPredicate): `H1-1`, `H1-1E` and `H1-1SC` all support a maximum of 20 qubits. `H2-1`, `H2-1E` and `H2-1SC` all support a maximum of 56 qubits.

# Job Statuses

When using the `QuantinuumBackend` to run circuits there are several possible circuit statuses.

- queued - The job has been queued but has not yet been run.
- running - The circuit is currently being run on the device/emulator.
- completed - The job has finished.
- failed -  The job has failed.
- cancelling - The job is in the process of being cancelled.
- cancelled - The job has been cancelled.

The status of the job can be checked with by using the `circuit_status` method. To cancel a job simply use the `cancel` method and supply the job handle as a parameter.

# Additional Backend Capabilities

The backend available through pytket-quantinuum has a `cost` method. This calculates the cost (in HQCs) required to execute the circuit for the specified number of shots.

Every backend also supports mid-circuit measurements and fast classical feedforward.

The `process_circuits` method for the QuantinuumBackend accepts the following additional keyword arguments.

- `postprocess` : boolean flag to allow classical postprocessing.
- `noisy_simulation` : boolean flag to specify whether the simulator should
  perform noisy simulation with an error model (default value is `True`).
- `group` : string identifier of a collection of jobs, can be used for usage tracking.

For the Quantinuum `Backend`, `process_circuits` returns a `ResultHandle` object containing a `job_id` and a postprocessing ( `ppcirc`) circuit if there is one.

The `logout()` method clears stored JSON web tokens and the user will have to sign in again to access the Quantinuum API.

## Persistent Authentication Token Storage

Following a successful login, the refresh token and the ID token, which are required for making further requests, will be saved.
This means you won't need to re-enter your credentials until these tokens expire. By default, these tokens are only stored in memory and will be removed once the Python session ends or if you manually log out.

For more persistent storage, consider using the `QuantinuumConfigCredentialStorage`. This storage option saves your username and the authentication tokens to the `pytket` configuration file, ensuring they persist beyond the current session.
To enable this, pass `QuantinuumConfigCredentialStorage` as an argument to `QuantinuumAPI`, which is then provided to `QuantinuumBackend`.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)
backend = QuantinuumBackend(
    device_name=machine,
    api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
)
backend.login() # username and tokens saved to the configuration file.
# A new QuantinuumAPI instance with QuantinuumConfigCredentialStorage
# will automatically load the credential from the configuration file.
backend2 = QuantinuumBackend(
    device_name=machine,
    api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
)
backend2.backend_info # No need to login again
```

Class methods use in-memory credential storage by default, so you need to explicitly set the `api_handler`:

```{code-cell} ipython3
---
tags: [skip-execution]
---
QuantinuumBackend.available_devices(
  api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage())
)
```

## Partial Results Retrieval

The {py:class}`QuantinuumBackend` also supports giving the user partial results from unfinished jobs.
This can be done as follows.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.quantinuum import QuantinuumBackend

# Submit circuit to QuantinuumBackend
backend = QuantinuumBackend('H1-1')
compiled_circ = backend.get_compiled_circuit(circ) # circ defined elsewhere
handle = backend.process_circuit(compiled_circ, n_shots=3000)

# Retrieve partial shots:counts from the handle of an unfinished job
partial_result, job_status = backend.get_partial_result(handle)
print(partial_result.get_counts())
```

This feature has a number of potential use cases.

Firstly partial results can be used to diagnose potential faults with submitted jobs.
For instance if the partial results look worse than expected this could indicate a bug in the submitted circuit(s). If this is the case the user may want to cancel the job to avoid using machine time and resubmit once the issue is resolved.
If partial results indicate that a job is taking longer run than anticipated then the user can cancel the job and consider redesigning their experiment.

Also partial results enable users to quickly validate basic execution for very large jobs which may take days to complete.

## Leakage Gadget Detection

When running circuits on the {py:class}`QuantinuumBackend`, one source of error is "leakage", where with some small probability a qubit will experience leakage into electronic states outside the qubit subspace. When this occurs, none of the remaining gates in the circuit will have any effect and so this leads to erroneous results.
Such leakage errors can be detected at the circuit level by running a special circuit gadget between a data qubit and an ancilla qubit. We can then discard shots where a leakage error is detected using {py:meth}`prune_shots_detected_as_leaky`.
For a more detailed explanation we refer to [Eliminating Leakage Errors in Hyperfine Qubits](https://arxiv.org/abs/1912.13131) by D. Hayes, D. Stack, B. Bjork, A. C. Potter, C. H. Baldwin and R. P. Stutz and the corresponding [notebook tutorial](https://github.com/CQCL/pytket-quantinuum/blob/develop/examples/Quantinuum_leakage_detection.ipynb).

## Batching

Quantinuum backends (except syntax checkers) support batching of jobs (circuits). To create
a batch of jobs, users submit the first job, then signal that subsequent jobs should
be added to the same batch using the handle of the first. The backend queue
management system will start the batch as soon as the first job reaches the
front of the queue and ensure subsequent batch jobs are run one after the other,
until the end of the batch is reached or there are no new jobs added to the batch
for ~1 min (at which point the batch expires and any subsequent jobs will be
added to the standard queue).

The standard `process_circuits` method **no
longer batches by default**. To use batching first start the batch with
`start_batch`, which has a similar interface to `process_circuit` but with
an extra first argument `max_batch_cost`:

```{code-cell} ipython3
---
tags: [skip-execution]
---
h1 = backend.start_batch(max_batch_cost=300, circuit=circuit, n_shots=100)
```

Add to the batch with subsequent calls of `add_to_batch` which takes as first
argument the handle of the first job of the batch, and has the optional keyword
argument `batch_end` to signal the end of a batch (default `False`).

```{code-cell} ipython3
---
tags: [skip-execution]
---
h2 = backend.add_to_batch(h1, circuit_2, n_shots=100)
h3 = backend.add_to_batch(h1, circuit_3, n_shots=100, batch_end=True)
```

The batch feature on Quantinuum systems gives users the ability to create "ad-hoc" reservations. Circuits submitted together in a batch will run at one time. The benefit to users is that once a batch hits the front of the queue, jobs in a batch will run uninterrupted until they are completed.

Once a batch is submitted, jobs can continue to be added to the batch, ending either when the user signifies the end of a batch or after 1 minute of inactivity. Batches cannot exceed the maximum limit of 500 H-System Quantum Credits (HQCs) total.

If the total HQCs for jobs in a batch hit this limit or a smaller limit set by the user, those jobs *will not be cancelled*. Instead, they will continue to run as regular jobs in the queue instead of as a batch.

# Local Emulators

If `pytket-quantinuum` is installed with the `pecos` option:

```
pip install pytket-quantinuum[pecos]
```

then it is possible to run circuits on an emulator running on the local machine
instead of using the remote emulator.

For example, the "H1-1" device would have a counterpart device called "H1-1LE".
Running circuits on this device would be similar to using the "H1-1" device or
the remote emulator ("H1-1E"), but would not incur any cost in HQCs (and for
small circuits would typically be faster).

Currently this emulation is noiseless, so if noisy emulation is required it is
still necessary to use the remote emulators (such as "H1-1E").

A few of the `QuantinuumBackend` methods (`submit_program()`, `cancel()`,
and `get_partial_result()`) are not available for local-emulator backends.

Normally using the local emulator requires one initial online API query to
retrieve the device information. To use it completely offline (with the caveat
that this relies on hard-coded assumptions about the available devices), you can
use the `QuantinuumAPIOffline` when constructing the backend:

```
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
api_offline = QuantinuumAPIOffline()
backend = QuantinuumBackend(device_name="H1-1LE", api_handler = api_offline)
```

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
