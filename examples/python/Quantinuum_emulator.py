# # Submitting to Quantinuum Emulators via pytket

# This notebook contains examples for running quantum circuits on Quantinuum's emulators.<br>
# We can use an emulator to get an idea of what a quantum device might output for our quantum circuit. This enables circuit debugging and optimization before running on a physical machine.<br>
# See the *Quantinuum Systems User Guide* on the Quantinuum User Portal for detailed information on each of the systems available and workflow information including job submission, queueing, and data retention.
# * [Basic Example](#example)
# * [Without Noise](#no-noise)
# * [Stabilizer Emulator](#stabilizer)

# ## Basic Example <a class="anchor" id="example"></a>

# ### Circuit Preparation

# Create your circuit via the pytket python library. For details on getting started with `pytket`, see pytket's [Getting Started](https://cqcl.github.io/tket/pytket/api/getting_started.html) page.

from pytket.circuit import Circuit, fresh_symbol
from pytket.circuit.display import render_circuit_jupyter

# Set up Bell Test
circuit = Circuit(2, name="Bell Test")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()
render_circuit_jupyter(circuit)

# ### Select Device

# *Available emulators:*
# - `H1-1E`, `H1-2E`: Device-specific Emulators for H1-1 and H1-2

# Login to the Quantinuum API using your credentials and check the device status.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-2E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

print(machine, "status:", QuantinuumBackend.device_state(device_name=machine))

# ### Circuit Compilation

# `pytket` includes many features for optimizing circuits. This includes reducing the number of gates where possible and resynthesizing circuits for a quantum computer's native gate set. See the `pytket` [User Manual](https://cqcl.github.io/pytket/manual/index.html) for more information on all the options that are available.<br>
# Here the circuit is compiled with `get_compiled_circuit`, which includes optimizing the gates and resynthesizing the circuit to Quantinuum's native gate set. The `optimisation_level` sets the level of optimization to perform during compilation, check pytket documentation for more details.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=1)
render_circuit_jupyter(compiled_circuit)

# ### Run the Circuit

# Now the circuit can be run on an emulator.

n_shots = 100
handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
print(handle)

# Check the status of the job.
status = backend.circuit_status(handle)
print(status)

# ### Retrieve Results

# Once a job's status returns completed, results can be returned using the `get_result` function.
result = backend.get_result(handle)
result

# ### Save Results

# It is recommended that users save job results as soon as jobs are completed due to the Quantinuum data retention policy.
import json

with open("pytket_emulator_example.json", "w") as file:
    json.dump(result.to_dict(), file)

# ### Analyze Results

# The result output is just like that of a quantum device.<br>
# The simulation by default runs with noise; therefore, it won't give back all `11`'s.
result = backend.get_result(handle)
print(result.get_distribution())

print(result.get_counts())

# ## Without Noise <a class="anchor" id="no-noise"></a>

# The Quantinuum emulators may be run with or without the physical device noise model. We can set `noisy_simulation=False` to do this.

n_shots = 100
no_error_model_handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, noisy_simulation=False
)
print(no_error_model_handle)

no_error_model_status = backend.circuit_status(no_error_model_handle)
print(no_error_model_status)

no_error_model_result = backend.get_result(no_error_model_handle)
no_error_model_result

with open("pytket_emulator_no_error_model_example.json", "w") as file:
    json.dump(result.to_dict(), file)

no_error_model_result = backend.get_result(no_error_model_handle)
print(no_error_model_result.get_distribution())
print(no_error_model_result.get_counts())

# ## Stabilizer Emulator <a class="anchor" id="stabilizer"></a>

# By default, emulations are run using a state-vector emulator, which simulates any quantum operation. However, if the quantum operations are all Clifford gates, it can be faster for complex circuits to use the `stabilizer` emulator. The stabilizer emulator is requested in the setup of the `QuantinuumBackend` with the `simulator` input option. This only applies to Quantinuum emulators.

machine = "H1-2E"
stabilizer_backend = QuantinuumBackend(device_name=machine, simulator="stabilizer")
print(machine, "status:", QuantinuumBackend.device_state(device_name=machine))
print("Simulation type:", stabilizer_backend.simulator_type)

n_shots = 100
stabilizer_handle = stabilizer_backend.process_circuit(
    compiled_circuit, n_shots=n_shots
)
print(stabilizer_handle)

stabilizer_status = stabilizer_backend.circuit_status(stabilizer_handle)
print(stabilizer_status)

stabilizer_result = stabilizer_backend.get_result(stabilizer_handle)
stabilizer_result

with open("pytket_emulator_stabilizer_example.json", "w") as file:
    json.dump(result.to_dict(), file)

stabilizer_result = stabilizer_backend.get_result(stabilizer_handle)
print(stabilizer_result.get_distribution())
print(stabilizer_result.get_counts())

# <div align="center"> &copy; 2022 by Quantinuum. All Rights Reserved. </div>
