# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Submitting to Quantinuum Emulators

# This notebook contains examples for running quantum circuits on Quantinuum's emulators via `pytket`.

# An emulator can be used to get an idea of what a quantum device will output for our quantum circuit. This enables circuit debugging and optimization before running on a physical machine. Emulators differ from simulators in that they model the physical and noise model of the device whereas simulators may model noise parameters, but not physical parameters. The Quantinuum emulators run on a physical noise model of the Quantinuum H-Series devices. There are various noise/error parameters modeled. For detailed information on the noise model, see the *Quantinuum System Model H1 Emulator Product Data Sheet* on the user portal.

# There are a few options for using the emulator:

# 1. **Basic Usage:** Use the emulator as provided, which represents both the physical operations in the device as well as the noise. This the most common and simplest way to use the emulator.
# 2. **Noiseless Emulation:** Use the emulator without the physical noise model applied. The physical device operations are represented, but all errors are set to 0.
# 3. **Noise Parameters (*advanced option*):** Experiment with the noise parameters in the emulator. There is no guarantee that results achieved changing these parameters will represent outputs from the actual quantum computer represented.
# 4. **Stabilizer Emulator:** Use of the emulator for circuits involving only Clifford operations.

# For more information, see the *Quantinuum System Model H1 Emulator Product Data Sheet*, *Quantinuum Systems User Guide*, and *Quantinuum Application Programming Interface (API)* on the Quantinuum User Portal for detailed information on each of the emulators available and workflow information including job submission, queueing, and the full list of options available.

# **Emulator Usage:**
# * [Basic Usage](#basic-usage)
# * [Noiseless Emulation](#no-noise)
# * [Noise Parameters (*advanced*)](#noise)
# * [Stabilizer Emulator](#stabilizer)
# ## Emulator Usage
# ### Basic Usage <a class="anchor" id="basic-usage"></a>

# This section covers usage of the emulator which represents a physical and noise model of the device being used. For example, if using the `H1-1E` target, this emulates the H1-1 quantum computer.

# Here the circuit is created via the pytket python library. For details on getting started with `pytket`, see pytket's [Getting Started](https://cqcl.github.io/tket/pytket/api/getting_started.html) page.

from pytket.circuit import Circuit
from pytket.circuit.display import render_circuit_jupyter

circuit = Circuit(2, name="Bell Test")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()

render_circuit_jupyter(circuit)

# Select the emulation device. See the *Quantinuum Systems User Guide* in the *Examples* tab on the *Quantinuum User Portal* for information and target names for each of the emulators available.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

print(machine, "status:", backend.device_state(device_name=machine))

# Compile the circuit to the Quantinuum backend with `get_compiled_circuit`. See the `pytket` [User Manual](https://cqcl.github.io/pytket/manual/index.html) for more information on all the options that are available.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=1)

render_circuit_jupyter(compiled_circuit)

# Check the circuit HQC cost before running on the emulator.

n_shots = 100
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")

# Run the circuit on the emulator chosen.

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
print(handle)

# Check the job status.

status = backend.circuit_status(handle)
print(status)

# Once a job's status returns completed, return results with the `get_result` function.

result = backend.get_result(handle)

# It is recommended to save job results as soon as jobs are completed due to the Quantinuum data retention policy.

import json

with open("pytket_emulator_example.json", "w") as file:
    json.dump(result.to_dict(), file)

# The result output is just like that of a quantum device. The simulation by default runs with noise.

result = backend.get_result(handle)
print(result.get_distribution())
print(result.get_counts())

# ### Noiseless Emulation <a class="anchor" id="no-noise"></a>

# The Quantinuum emulators may be run with or without the physical device's noise model. The default is the emulator runs with the physical noise model turned on. The physical noise model can be turned off by setting `noisy_simulation=False`.

n_shots = 100
no_error_model_handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, noisy_simulation=False
)
print(no_error_model_handle)

no_error_model_status = backend.circuit_status(no_error_model_handle)
print(no_error_model_status)

no_error_model_result = backend.get_result(no_error_model_handle)

with open("pytket_emulator_noiseless_example.json", "w") as file:
    json.dump(result.to_dict(), file)

no_error_model_result = backend.get_result(no_error_model_handle)

print(no_error_model_result.get_distribution())

print(no_error_model_result.get_counts())

# ### Noise Parameters <a class="anchor" id="noise"></a>

# The emulator runs with default error parameters that represent a noise environment similar to the physical devices. The `error-params` option can be used to override these error parameters and do finer-grain tweaks of the error model. For detailed information on the noise model, see the *Quantinuum System Model H1 Emulator Product Data Sheet* or the *Quantinuum Application Programming Interface (API)*.

# In this section, examples are given for experimenting with the noise and error parameters of the emulators. These are advanced options and not recommended to start with when doing initial experiments. As mentioned above, there is no guarantee that results achieved changing these parameters will represent outputs from the actual quantum computer represented.

# **Note**: All the noise parameters are used together any time a simulation is run. If only some of the parameters are specified, the rest of the parameters are used at their default settings. The parameters to override are specified with the `options` parameter.

# * [Physical Noise](#physical-noise)
# * [Dephasing Noise](#dephasing-noise)
# * [Arbitrary Angle Noise Scaling](#arbitrary-angle-noise)
# * [Scaling](#scaling)

# #### Physical Noise <a class="anchor" id="physical-noise"></a>

# See the *Quantinuum System Model H1 Emulator Product Data Sheet* on the Quantinuum user portal for information on these parameters.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "error-params": {
                "p1": 4e-5,
                "p2": 3e-3,
                "p_meas": 3e-3,
                "p_init": 4e-5,
                "p_crosstalk_meas": 1e-5,
                "p_crosstalk_init": 3e-5,
                "p1_emission_ratio": 0.15,
                "p2_emission_ratio": 0.3,
            }
        }
    },
)
result = backend.get_result(handle)

print(result.get_distribution())

# #### Dephasing Noise <a class="anchor" id="dephasing-noise"></a>

# See the *Quantinuum System Model H1 Emulator Product Data Sheet* on the user portal for information on these parameters.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "error-params": {
                "quadratic_dephasing_rate": 0.2,
                "linear_dephasing_rate": 0.3,
                "coherent_to_incoherent_factor": 2.0,
                "coherent_dephasing": False,  # False => run the incoherent noise model
                "transport_dephasing": False,  # False => turn off transport dephasing error
                "idle_dephasing": False,  # False => turn off idle dephasing error
            },
        }
    },
)
result = backend.get_result(handle)

print(result.get_distribution())

# #### Arbitrary Angle Noise Scaling <a class="anchor" id="arbitrary-angle-noise"></a>

# See the *Quantinuum System Model H1 Emulator Product Data Sheet* on the user portal for information on these parameters.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "error-params": {
                "przz_a": 1.09,
                "przz_b": 0.035,
                "przz_c": 1.09,
                "przz_d": 0.035,
                "przz_power": 1 / 2,
            },
        }
    },
)
result = backend.get_result(handle)

print(result.get_distribution())

# #### Scaling <a class="anchor" id="scaling"></a>

# All the error rates can be scaled linearly using the `scale` parameter. See the *Quantinuum System Model H1 Emulator Product Data Sheet* on the user portal for more information.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "error-params": {
                "scale": 0.1,  # scale error rates linearly by 0.1
            },
        }
    },
)
result = backend.get_result(handle)

print(result.get_distribution())

# Other aspects of the noise model can scale specific error rates in the error model, which are modeled here.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "error-params": {
                "p1_scale": 0.1,
                "p2_scale": 0.1,
                "meas_scale": 0.1,
                "init_scale": 0.1,
                "memory_scale": 0.1,
                "emission_scale": 0.1,
                "crosstalk_scale": 0.1,
                "leakage_scale": 0.1,
            },
        }
    },
)
result = backend.get_result(handle)

print(result.get_distribution())

# ### Stabilizer Emulator <a class="anchor" id="stabilizer"></a>

# By default, emulations are run using a state-vector emulator, which simulates any quantum operation. However, if the quantum operations are all Clifford gates, it can be faster for complex circuits to use the `stabilizer` emulator. The stabilizer emulator is requested in the setup of the `QuantinuumBackend` with the `simulator` input option. This only applies to Quantinuum emulators.

machine = "H1-1E"

stabilizer_backend = QuantinuumBackend(device_name=machine, simulator="stabilizer")

print(machine, "status:", stabilizer_backend.device_state(device_name=machine))
print("Simulation type:", stabilizer_backend.simulator_type)

n_shots = 100
stabilizer_handle = stabilizer_backend.process_circuit(
    compiled_circuit, n_shots=n_shots
)
print(stabilizer_handle)

stabilizer_status = stabilizer_backend.circuit_status(stabilizer_handle)
print(stabilizer_status)

stabilizer_result = stabilizer_backend.get_result(stabilizer_handle)
with open("pytket_emulator_stabilizer_example.json", "w") as file:
    json.dump(result.to_dict(), file)

stabilizer_result = stabilizer_backend.get_result(stabilizer_handle)

print(stabilizer_result.get_distribution())

print(stabilizer_result.get_counts())

# #### Noiseless Stabilizer

# A noiseless stabilizer simulation can be specified via options in the `process_circuit` function with the following options:

# - `simulator`: choose to run with a `stabilizer` simulator or `state-vector` (default is `state-vector`)
# - `error-model`: whether to run with or without the physical device noise model on or off. The default is `True`, which means the physical noise model is turned on. If set to `False`, the physical noise model is turned off, performing noiseless simulation.

handle = backend.process_circuit(
    compiled_circuit,
    n_shots=100,
    request_options={
        "options": {
            "simulator": "stabilizer",
            "error-model": False,
        }
    },
)
result = backend.get_result(handle)
print(result.get_distribution())

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
