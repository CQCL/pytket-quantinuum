# # Quantinuum Circuit Submissions via pytket

# This notebook contains basic circuit submission examples to Quantinuum quantum hardware via `pytket`.

# * [What is TKET?]((#tket)
# * [Step by Step](#step-by-step)
#     * [Circuit Preparation](#circuit-preparation)
#     * [Select Device](#select-device)
#     * [Circuit Compilation](#circuit-compilation)
#     * [Circuit Cost](#circuit-cost)
#     * [Run the Circuit](#run-circuit)
#     * [Retrieve Results](#retrieve-results)
#     * [Save Results](#save-results)
#     * [Analyze Results](#analyze-results)
#     * [Cancel Jobs](#cancel-jobs)
# * [Additional Features](#additional-features)
#     * [Extended Circuit Compilation](#circuit-compilation-extended)
#     * [Batch Submission](#batch-submission)
#     * [Parametrized Circuits](#parametrized-circuits)
#     * [Conditional Gates](#conditional-gates)

# ## What is TKET? <a class="anchor" id="step-by-step"></a>

# The TKET framework (pronounced "ticket") is a software platform for the development and execution of gate-level quantum computation, providing state-of-the-art performance in circuit compilation. It was created and is maintained by Quantinuum. The toolset is designed to extract the most out of the available NISQ devices of today and is platform-agnostic.

# In python, the `pytket` packages is available for python 3.8+. The `pytket` and `pytket-quantinuum` packages are included as part of the installation instructions on the user portal.

# For more information on TKET, see the following links:
# - [TKET user manual](https://cqcl.github.io/pytket/manual/manual_intro.html)
# - [TKET overview and demo video](https://www.youtube.com/watch?v=yXKSpvgAtrk)
# - [Quantum Compilation with TKET](https://calmaccq.github.io/tket_blog/tket_compilation.html)

# This notebook covers how to use `pytket` in conjunction with `pytket-quantinuum` to submit to Quantinuum devices. The quantum compilation step is demonstrated, but for a full overview of quantum compilation with TKET, the last link above is recommended.

# See the links below for the `pytket` and `pytket-quantinuum` documentation:
# - [pytket](https://cqcl.github.io/tket/pytket/api/index.html)
# - [pytket-quantinuum](https://cqcl.github.io/pytket-quantinuum/api/index.html)

# ## Step by Step <a class="anchor" id="step-by-step"></a>

# ### Circuit Preparation <a class="anchor" id="circuit-preparation"></a>

# Create your circuit via the pytket python library. For details on getting started with `pytket`, see pytket's [Getting Started](https://cqcl.github.io/tket/pytket/api/getting_started.html) page.
from pytket.circuit import Circuit, fresh_symbol
from pytket.circuit.display import render_circuit_jupyter

# Set up Bell Test
circuit = Circuit(2, name="Bell Test")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()

render_circuit_jupyter(circuit)

# ### Select Device <a class="anchor" id="select-device"></a>

# Select a machine and login to the Quantinuum API using your credentials. See the *Quantinuum Systems User Guide* in the *Examples* tab on the *Quantinuum User Portal* for information and target names for each of the H-Series systems available.

# Users need to login once per session. In the notebook, a dialogue box will ask for credentials. If running a script, users be prompted at the shell. You can also [save your email in the pytket config](https://cqcl.github.io/tket/pytket/api/config.html?highlight=pytket%20config#module-pytket.config).

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-2E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

# The device status can be checked using `device_state`.

print(machine, "status:", QuantinuumBackend.device_state(device_name=machine))

# Available devices can be viewed using the `available_devices` function. Additional information is returned, here just the device names are pulled in.

[x.device_name for x in QuantinuumBackend.available_devices()]

# ### Circuit Compilation <a class="anchor" id="circuit-compilation"></a>

# Circuits submitted to Quantinuum H-Series quantum computers and emulators are automatically run through TKET compilation passes for H-Series hardware. This enables circuits to be automatically optimized for H-Series systems and run more efficiently.

# More information on the specific compilation passes applied can be found on the `pytket-quantinuum` documentation, specifically the [Default Compilation](https://cqcl.github.io/pytket-quantinuum/api/index.html#default-compilation) section. In the H-Series software stack, the optimization level applied is set with the `tket-opt-level` parameter. **The default compilation setting for circuits submited to H-Series sytems is optimization level 2.** More information is found in the *Quantinuum Application Programming Interface (API) Specification*.

# When using `pytket` before submitting to hardware, the `get_compiled_circuit` function performs the same compilation passes run after submission to Quantinuum systems. The advantage of using the function before submitting to H-Series hardware is to see exactly what circuit optimizations will be performed when submitted to hardware and determine if a different optimization level is desired. The `optimisation_level` parameter in the `get_compiled_circuit` function corresponds directly to the the level of optimisation after submitting to the H-Series systems and to the `tket-opt-level` parameter in the H-Series API. The default compilation for the `get_compiled_circuit` function is optimization level 2, the same as when submitting to the H-Series directly.

# Since the TKET compilation passes have been integrated into the H-Series stack, performing circuit optimizations is redundant before submitting to hardware, unless the user would like to see the optimizations applied before submitting. Given this, users may take 1 of 3 approaches when submitting jobs:
# 1. Use `optimisation_level=0` when running `get_compiled_circuit`, then submit the circuit using `process_circuits` knowing that the corresponding optimization level actually run will be 2.
# 2. Use the `get_compiled_circuit` function with the desired optimization level to observe the transformed circuit before submitting and then specify `tket-opt-level=None` in the `process_circuits` function when submitting, in order for the optimizations to be applied as desired.
# 3. If the user desires to have no optimizations applied, use `optimisation_level=0` in `get_compiled_circuit` and `tket-opt-level=None` in `process_circuits`. This should be specified in both functions.

# In this example, option 1 is illustrated, using `get_compiled_circuit` just to rebase the circuit and leaving the optimizations to be done in the H-Series stack.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

render_circuit_jupyter(compiled_circuit)

# ### Circuit Cost <a class="anchor" id="circuit-cost"></a>

# Before running on Quantinuum systems, it is good practice to check how many HQCs a job will cost, in order to plan usage. In `pytket` this can be done using the `cost` function of the `QuantinuumBackend`.

# Note that in this case because an emulator is used, the specific syntax checker the emulator uses is specified. This is an optional parameter not needed if you are using a quantum computer target.

n_shots = 100
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-2SC")

# ### Run the Circuit <a class="anchor" id="run-circuit"></a>

# Now the circuit can be run on Quantinuum systems.

# **Note:** As described above, the TKET compilation optimization level 2 will be applied since no `tket-opt-level` is specified.

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
print(handle)

# The status of a submitted job can be viewed at any time, indicating if a job is in the queue or completed. Additional information is also provided, such as queue position, start times, completion time, and circuit cost in H-Series Quantum Credits (HQCs).

status = backend.circuit_status(handle)
print(status)

# ### Retrieve Results <a class="anchor" id="retrieve-results"></a>

# Once a job's status returns completed, results can be returned using the `get_result` function.

result = backend.get_result(handle)

# For large jobs, there is also the ability to return partial results for unfinished jobs. For more information on this feature, see [Partial Results Retrieval](https://cqcl.github.io/pytket-quantinuum/api/#partial-results-retrieval).

partial_result, job_status = backend.get_partial_result(handle)

print(partial_result.get_counts())

# ### Save Results <a class="anchor" id="save-results"></a>

# It is recommended that users save job results as soon as jobs are completed due to the Quantinuum data retention policy.

import json

with open("pytket_example.json", "w") as file:
    json.dump(result.to_dict(), file)

# Results can be loaded to their original format using `BackendResult.from_dict`.

from pytket.backends.backendresult import BackendResult

with open("pytket_example.json") as file:
    data = json.load(file)

result = BackendResult.from_dict(data)

# ### Analyze Results <a class="anchor" id="analyze-results"></a>

# There are multiple options for analyzing results with pytket. A few examples are highlighted here. More can be seen at [Interpreting Results](https://cqcl.github.io/pytket/manual/manual_backend.html#interpreting-results).

result = backend.get_result(handle)
print(result.get_distribution())

print(result.get_counts())

# ### Canceling jobs <a class="anchor" id="cancel-jobs"></a>

# Jobs that have been submitted can also be cancelled if needed.

# backend.cancel(handle)

# ## Additional Features <a class="anchor" id="additional-features"></a>

# This section covers additional features available in `pytket`.

# * [Extended Circuit Compilation](#circuit-compilation-extended)
# * [Batch Submission](#batch-submission)
# * [Parametrized Circuits](#parametrized-circuits)
# * [Conditional Gates](#conditional-gates)

# ### Extended Circuit Compilation <a class="anchor" id="circuit-compilation-extended"></a>

# This section leverages the discussion in the [Circuit Compilation](#circuit-compilation) section to illsutrate how to turn TKET compilations on or off in the `process_circuit` function, specifically for options 2 and 3.

# For option 2 as described in [Circuit Compilation](#circuit-compilation), suppose a user explores the results of TKET compilation passes on a circuit and finds that `optimisation_level=1` is desirable. The submission below specifies this in the `get_compiled_circuit` function with optimization level 1. Because the circuit is optimized beforehand, the TKET optimization in the H-Series stack should be turned off. The value `tket-opt-level:None` turns off TKET optimization in the H-Series stack.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=1)

handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, options={"tket-opt-level": None}
)
print(handle)

# For option 3 as described in [Circuit Compilation](#circuit-compilation), suppose a user wants to turn off all optimizations in the stack, even simple single-qubit combinations done by the H-Series compiler. This can be done by setting `optimisation_level=0` in `get_compiled_circuit` and setting `tket-opt-level:None` in the `process_circuits` function.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, options={"tket-opt-level": None}
)
print(handle)

# ### Batch Submission <a class="anchor" id="batch-submission"></a>

# The batch feature on Quantinuum systems gives users the ability to create "ad-hoc" reservations. Circuits submitted together in a batch will run at one time. The benefit to users is that once a batch hits the front of the queue, jobs in a batch will run uninterrupted until they are completed.

# Once a batch is submitted, jobs can continue to be added to the batch, ending either when the user signifies the end of a batch or after 1 minute of inactivity.

# Batches cannot exceed the maximum limit of 2,000 H-System Quantum Credits (HQCs) total. If the total HQCs for jobs in a batch hit this limit or a smaller limit set by the user, those jobs *will not be cancelled*. Instead, they will continue to run as regular jobs in the queue instead of as a batch.

# Currently only the quantum computer and emulator targets support the batching feature. Batching is not supported on the syntax checkers.

# For more information on using this feature in `pytket-quantinuum`, see [Batching](https://cqcl.github.io/pytket-quantinuum/api/index.html#batching).

# To start a batch, use the `start_batch` function, specifying the `max_batch_cost` in HQCs to enforce.

machine = "H1-1E"
n_shots = 100
max_batch_cost = 100

backend = QuantinuumBackend(device_name=machine)

batch1 = backend.start_batch(
    max_batch_cost=max_batch_cost, circuit=compiled_circuit, n_shots=n_shots
)

# Additional jobs can be added to the batch using the `add_to_batch` function. The end of a batch can optionally be specified with the `batch_end` flag.

batch2 = backend.add_to_batch(batch1, compiled_circuit, n_shots=n_shots)
batch3 = backend.add_to_batch(batch1, compiled_circuit, n_shots=n_shots, batch_end=True)

# The status for the batch jobs can be checked once submitted.

handle_list = [batch1, batch2, batch3]

status_list = [backend.circuit_status(h) for h in handle_list]

status_list

# Results for batch submissions can be returned using `get_results` (note the plural).

results = backend.get_results(handle_list)

for result in results:
    print(result.get_counts())

# ### Parametrized Circuits <a class="anchor" id="parametrized-circuits"></a>

# Parametrized circuits are common in variational algorithms. Pytket supports parameters within circuits via symbols. For more information, see [Symbolic Circuits](https://cqcl.github.io/pytket/manual/manual_circuit.html?highlight=paramet#symbolic-circuits).

from pytket.circuit import fresh_symbol

# Set up parametrized circuit
a = fresh_symbol("a")
circuit = Circuit(3, name="Parametrized Circuit")
circuit.X(0)
circuit.CX(0, 1).CX(1, 2)
circuit.Rz(a, 2)
circuit.CX(1, 2).CX(0, 1)

render_circuit_jupyter(circuit)

# Note the substitution of an actual value to the `a` variable below.

# Create a version of the circuit that utilizes a specific value for the variable a
simulation_circuit = circuit.copy()
simulation_circuit.measure_all()
simulation_circuit.symbol_substitution({a: -0.09})

# Compile the circuit: this includes optimizing the gates and resynthesizing the circuit to Quantinuum's native gate set
compiled_circuit = backend.get_compiled_circuit(simulation_circuit)

render_circuit_jupyter(compiled_circuit)

n_shots = 100
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-2SC")

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)

status = backend.circuit_status(handle)
print(status)

result = backend.get_result(handle)

# ### Conditional Gates <a class="anchor" id="conditional-gates"></a>

# Pytket supports conditional gates. This may be for implementing error correction or reducing noise. This capability is well-supported by Quantinuum hardware, which supports mid-circuit measurement and qubit reuse. See [Conditional Gates](https://cqcl.github.io/pytket/manual/manual_circuit.html#classical-and-conditional-operations) for more information on pytket's implementation. The following example demonstrates the quantum teleportation protocol.

from pytket.circuit import Circuit, if_bit

# create a circuit and add quantum and classical registers
circ = Circuit(name="Conditional Gates Example")
qreg = circ.add_q_register("q", 3)
creg = circ.add_c_register("b", 2)

# prepare q[0] to be in the state |->, which we wish to teleport to q[2]
circ.X(qreg[0]).H(qreg[0])

# prepare a Bell state on qubits q[1] and q[2]
circ.H(qreg[1])
circ.CX(qreg[1], qreg[2])

# construct the teleportation protocol
circ.CX(qreg[0], qreg[1])
circ.H(qreg[0])
circ.Measure(qreg[0], creg[0])
circ.Measure(qreg[1], creg[1])

# if (creg[1] == 1)
circ.X(qreg[2], condition=if_bit(creg[1]))

# if (creg[0] == 1)
circ.Z(qreg[2], condition=if_bit(creg[0]))

render_circuit_jupyter(circ)

# We can utilise pytket's [Assertion](https://cqcl.github.io/pytket/manual/manual_assertion.html#assertion) feature to verify the successful teleportation of the state $| - \rangle$.

from pytket.circuit import ProjectorAssertionBox
import numpy as np

# |-><-|
proj = np.array([[0.5, -0.5], [-0.5, 0.5]])
circ.add_assertion(ProjectorAssertionBox(proj), [qreg[2]], name="debug")

render_circuit_jupyter(circ)

machine = "H1-2E"
n_shots = 100
backend = QuantinuumBackend(device_name=machine)
compiled_circuit = backend.get_compiled_circuit(circ)
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-2SC")

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)

status = backend.circuit_status(handle)
status

result = backend.get_result(handle)

# The `get_debug_info` function returns the success rate of the state assertion averaged across shots. Note that the failed shots are caused by the simulated device errors

result.get_debug_info()

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
