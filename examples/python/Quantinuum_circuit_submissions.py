# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # How to Submit Quantum Circuits to H-Series Backends

# This notebook contains basic circuit submission examples to Quantinuum quantum hardware via `pytket`.

# * [What is TKET?](#What-is-TKET?)
# * [Step by Step](#step-by-step)
#     * [Circuit Preparation](#Circuit-Preparation)
#     * [Select Device](#Select-Device)
#     * [Circuit Compilation](#Circuit-Compilation)
#     * [Circuit Cost](#Circuit-Cost)
#     * [Run the Circuit](#Run-the-Circuit)
#     * [Retrieve Results](#Retrieve-Results)
#     * [Save Results](#Save-Results)
#     * [Analyze Results](#Analyze-Results)
#     * [Cancel Jobs](#Cancel-Jobs)
# ## What is TKET?

# The TKET framework (pronounced "ticket") is a software platform for the development and execution of gate-level quantum computation, providing state-of-the-art performance in circuit compilation. It was created and is maintained by Quantinuum. The toolset is designed to extract the most out of the available NISQ devices of today and is platform-agnostic.

# In python, the `pytket` packages is available for python 3.9+. The `pytket` and `pytket-quantinuum` packages are included as part of the installation instructions on the user portal.

# For more information on TKET, see the following links:
# - [TKET user manual](https://tket.quantinuum.com/user-manual/manual_intro.html)
# - [TKET notebook examples](https://tket.quantinuum.com/examples/)

# This notebook covers how to use `pytket` in conjunction with `pytket-quantinuum` to submit to Quantinuum devices. The quantum compilation step is demonstrated, but for a full overview of quantum compilation with TKET, the last link above is recommended.

# See the links below for the `pytket` and `pytket-quantinuum` documentation:
# - [pytket](https://tket.quantinuum.com/api-docs/)
# - [pytket-quantinuum](https://tket.quantinuum.com/extensions/pytket-quantinuum/)
# ## Step by Step

# ### Circuit Preparation

# Create your circuit via the pytket python library. For details on getting started with `pytket`, see pytket's [Getting Started](https://tket.quantinuum.com/examples/Getting_started.html) page.

from pytket.circuit import Circuit, fresh_symbol
from pytket.circuit.display import render_circuit_jupyter

circuit = Circuit(2, name="Bell Test")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()

render_circuit_jupyter(circuit)

# ### Select Device

# Select a machine and login to the Quantinuum API using your credentials. See the *Quantinuum Systems User Guide* in the *Examples* tab on the *Quantinuum User Portal* for information and target names for each of the H-Series systems available.

# Users need to login once per session. In the notebook, a dialogue box will ask for credentials. If running a script, users be prompted at the shell. You can also [save your email in the pytket config](https://tket.quantinuum.com/api-docs/config.html).

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

# The device status can be checked using `device_state`.

print(machine, "status:", QuantinuumBackend.device_state(device_name=machine))

# Available devices can be viewed using the `available_devices` function. Additional information is returned, here just the device names are pulled in.

[x.device_name for x in QuantinuumBackend.available_devices()]

# ### Circuit Compilation

# Circuits submitted to Quantinuum H-Series quantum computers and emulators are automatically run through TKET compilation passes for H-Series hardware. This enables circuits to be automatically optimized for H-Series systems and run more efficiently.

# **Note:** See the *Circuit Compilation for H-Series* notebook for detailed information about TKET compilation in the stack.

# In this example, optimisation level 0 is illustrated, using `get_compiled_circuit` just to rebase the circuit and leaving the optimizations to be done in the H-Series stack.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

render_circuit_jupyter(compiled_circuit)

# ### Circuit Cost

# Before running on Quantinuum systems, it is good practice to check how many HQCs a job will cost, in order to plan usage. In `pytket` this can be done using the `cost` function of the `QuantinuumBackend`.

# Note that in this case because an emulator is used, the specific syntax checker the emulator uses is specified. This is an optional parameter not needed if you are using a quantum computer target.

n_shots = 100
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")

# ### Run the Circuit

# Now the circuit can be run on Quantinuum systems.

# **Note:** As described above, the TKET compilation optimization level 2 will be applied since no `tket-opt-level` is specified.

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
print(handle)

# If you have a long-running job, you may want to close your session and load the job results later. To save the pytket job handle, run the following. The handle can be imported later.

import json

with open("pytket_example_job_handle.json", "w") as file:
    json.dump(str(handle), file)

# The status of a submitted job can be viewed at any time, indicating if a job is in the queue or completed. Additional information is also provided, such as queue position, start times, completion time, and circuit cost in H-Series Quantum Credits (HQCs).
status = backend.circuit_status(handle)
print(status)

# ### Retrieve Results

# Once a job's status returns completed, results can be returned using the `get_result` function. If you ran your job in a previous session and wnat to reload it, run the following.

from pytket.backends import ResultHandle

with open("pytket_example_job_handle.json") as file:
    handle_str = json.load(file)

handle = ResultHandle.from_str(handle_str)
result = backend.get_result(handle)

# For large jobs, there is also the ability to return partial results for unfinished jobs. For more information on this feature, see [Partial Results Retrieval](https://tket.quantinuum.com/extensions/pytket-quantinuum/api/#partial-results-retrieval).

partial_result, job_status = backend.get_partial_result(handle)

print(partial_result.get_counts())

# ### Save Results

# It is recommended that users save job results as soon as jobs are completed due to the Quantinuum data retention policy.

import json

with open("pytket_example.json", "w") as file:
    json.dump(result.to_dict(), file)

# Results can be loaded to their original format using `BackendResult.from_dict`.

from pytket.backends.backendresult import BackendResult

with open("pytket_example.json") as file:
    data = json.load(file)

result = BackendResult.from_dict(data)

# ### Analyze Results

# There are multiple options for analyzing results with pytket. A few examples are highlighted here. More can be seen at [Interpreting Results](https://tket.quantinuum.com/user-manual/manual_backend.html#interpreting-results).

result = backend.get_result(handle)
print(result.get_distribution())
print(result.get_counts())

# ### Canceling jobs

# Jobs that have been submitted can also be cancelled if needed.

# backend.cancel(handle)

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
