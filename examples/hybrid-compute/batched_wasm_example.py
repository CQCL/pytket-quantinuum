#!/usr/bin/env python
# coding: utf-8

# # Quantinuum Hybrid Compute via pytket
#
# This notebook contains an example of how to create and run hybrid classical-quantum workflows within the Quantinuum stack via `pytket`.
#
# **Note:** This notebook uses existing Wasm examples provided in this folder's subdirectories. To create and run your own Wasm functions, you'll need to set up an environment on your machine to support this workflow. Instructions for how to do this are given in this folder's README.
#
# * [Repeat Until Success](#repeat-until-success)

# ## Repeat Until Success <a class="anchor" id="repeat-until-success"></a>
#
# This is a simple example made of how you can process measurements, using classical logic via Wasm, to implement a quantum loop, or a repeat-until-success (RUS) style circuit, also called an exit-on-failure circuit.
#
# * [1 RUS Circuit](#1-rus-circuit)
# * [RUS Experiment](#rus-experiment)

# ### 1 RUS Circuit <a class="anchor" id="1-rus-circuit"></a>
#
# #### Import libraries
#
# Note the `wasm` module imported from `pytket` as well as several several conditional operators that are options. More information on the conditional operations available can be found in the user manual at [Classical and Conditional Operations](https://cqcl.github.io/pytket/manual/manual_circuit.html#classical-and-conditional-operations).

# #### Select Device
#
# Select device and login to the Quantinuum API using your credentials.


from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine)
backend.login()


# #### Set up Circuit
#
# Set up the RUS circuit. This involves 2 steps:
# 1. Set up Wasm File handler. This checks the Wasm file and can list the available functions within the file.
# 2. Set up the pytket circuit, which calls the Wasm file handler.


import pathlib
from pytket import wasm

rus_dir = pathlib.Path().cwd().joinpath("repeat_until_success")
wasm_file = rus_dir.joinpath("rus_wasm.wasm")
wfh = wasm.WasmFileHandler(wasm_file)

print(repr(wfh))


from pytket.circuit import Circuit, Qubit
from pytket.circuit.logic_exp import reg_lt


def build_rus_circuit(n_repetitions: int, cond_execute: int) -> Circuit:
    """
    n_repetitions (int): number of attempts
    cond_execute: condition to execute, i.e. run this until a total of cond_execute |1> measurements is achieved.

    """
    # Create circuit with two qubits
    circuit = Circuit(2, name=f"RUS_limit={n_repetitions}_cond={cond_execute}")

    # Add classical registers
    creg0 = circuit.add_c_register("creg0", 1)
    creg1 = circuit.add_c_register("creg1", 1)
    cond = circuit.add_c_register("cond", 32)
    count = circuit.add_c_register("count", 32)

    # Set cond to 0
    circuit.add_c_setreg(0, cond)

    # Loops
    for loop_iter in range(1, n_repetitions + 1):

        circuit.H(0, condition=reg_lt(cond, cond_execute))
        circuit.CX(0, 1, condition=reg_lt(cond, cond_execute))
        circuit.Measure(Qubit(1), creg1[0], condition=reg_lt(cond, cond_execute))

        # Add wasm call with the parameters creg1 and count, writing the result to cond
        # The function "add_count" is used from the Wasm file
        circuit.add_wasm_to_reg(
            "add_count",
            wfh,
            [creg1, count],
            [cond],
            condition=reg_lt(cond, cond_execute),
        )

        circuit.add_c_setreg(loop_iter, count, condition=reg_lt(cond, cond_execute))
        circuit.Reset(0, condition=reg_lt(cond, cond_execute))

    circuit.Measure(Qubit(0), creg0[0])
    circuit.Measure(Qubit(1), creg1[0])

    return circuit


# Set up the circuit.

from pytket.circuit.display import render_circuit_jupyter, view_browser

circuit = build_rus_circuit(5, 3)

# render_circuit_jupyter(circuit)
view_browser(circuit)


# #### Circuit Compilation
#
# `pytket` includes many features for optimizing circuits. This includes reducing the number of gates where possible and resynthesizing circuits for a quantum computer's native gate set. See the `pytket` [User Manual](https://cqcl.github.io/pytket/manual/index.html) for more information on all the options that are available.
#
# Here the circuit is compiled with `get_compiled_circuit`, which includes optimizing the gates and resynthesizing the circuit to Quantinuum's native gate set. The `optimisation_level` sets the level of optimisation to perform during compilation, check `pytket-quantinuum` documentation for more information, specifically the [Default Compilation](https://cqcl.github.io/pytket-quantinuum/api/index.html#default-compilation) section.


compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

view_browser(compiled_circuit)


# #### Check Circuit Cost
#
# Check the cost of the experiment before running.


n_shots = 1000
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")


# #### Submit and Run the Circuit
#
# Note the `wasm_file_handler` input in the `process_circuit` function. This ensures the Wasm functions are passed onto the Quantinuum API.


handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, wasm_file_handler=wfh
)
print(handle)


status = backend.circuit_status(handle)
print(status)


result = backend.get_result(handle)


print(result)


import json

with open(compiled_circuit.name + ".json", "w", encoding="utf-8") as file:
    json.dump(result.to_dict(), file)


# #### Analyze Results
#
# Define a function to analyze the counts of 0 and 1 measurements on the `creg1` register.

from pytket.unit_id import Bit


def analyze_rus_experiment(result: dict):
    """Analyze results of RUS experiment.

    result: (dict)
    """
    creg1_0_cnts = {}
    creg1_1_cnts = {}

    for key, res in result.items():
        # Retrieve which c_bit corresponds to creg1 for this result
        creg1 = [x for x in res.c_bits.keys() if x.reg_name == "creg1"][0]

        # Retrieve the creg1 results for that experiment
        result_creg1 = [x[res.c_bits[creg1]] for x in res.get_shots()]

        # Retrieve n_shots for that experiment
        n_shots = len(res.get_shots())

        # Retrieve limit and cond for that experiment
        limit = int(key[key.find("limit=") + 6 : key.find("_cond=")])
        cond = int(key[key.find("_cond=") + 6 :])

        # Store results
        creg1_0_cnts[limit, cond] = [
            result_creg1.count(0),
            result_creg1.count(0) / n_shots,
        ]
        creg1_1_cnts[limit, cond] = [
            result_creg1.count(1),
            result_creg1.count(1) / n_shots,
        ]

    return creg1_0_cnts, creg1_1_cnts


result1 = {}
result1[compiled_circuit.name] = result
creg1_0_cnts, creg1_1_cnts = analyze_rus_experiment(result1)
print(analyze_rus_experiment(result1))

# Results of 0 counts on the `creg1` register.


print(creg1_0_cnts)


# ### RUS Experiment <a class="anchor" id="rus-experiment"></a>
#
# A more interesting experiment is to experiment with what happens with different values of the `limit`. The function below submits a set of RUS experiments using the `pytket-quantinuum` batch feature.


def get_rus_circuit_batch(
    machine_name: str, limit_start: int, limit_end: int, cond_execute: int
) -> list[Circuit]:
    backend = QuantinuumBackend(device_name=machine_name)

    circuit_list = []

    for limit in range(limit_start, limit_end + 1):
        rus_circuit = build_rus_circuit(limit, cond_execute)
        compiled_rus_circuit = backend.get_compiled_circuit(rus_circuit)
        circuit_list.append(compiled_rus_circuit)

    return circuit_list


def get_batch_cost(
    compiled_circuit_list: list[Circuit], machine: str, n_shots: int
) -> float:

    syntax_checker = machine.replace("E", "SC")

    total_cost = 0
    for compiled_circuit in compiled_circuit_list:
        circuit_cost = backend.cost(
            circuit=compiled_circuit, syntax_checker=syntax_checker, n_shots=n_shots
        )
        if circuit_cost is not None:
            total_cost += circuit_cost

    return total_cost


def run_rus_experiment(
    machine: str,
    n_shots: int,
    max_batch_cost: int,
    cond_execute: int,
    limit_start: int,
    limit_end: int,
) -> tuple[dict, float]:

    backend = QuantinuumBackend(device_name=machine)
    rus_circuit_list = get_rus_circuit_batch(
        machine, limit_start, limit_end, cond_execute
    )

    experiment_cost: float = get_batch_cost(rus_circuit_list, machine, n_shots)

    batch_handles = {}

    batch_start = backend.start_batch(
        max_batch_cost,
        rus_circuit_list[0],
        n_shots,
        wasm_file_handler=wfh,
    )
    batch_handles[rus_circuit_list[0].name] = batch_start

    for compiled_circuit in rus_circuit_list[1:-2]:
        batch_mid = backend.add_to_batch(
            batch_start,
            compiled_circuit,
            n_shots=n_shots,
            wasm_file_handler=wfh,
        )
        batch_handles[compiled_circuit.name] = batch_mid

    batch_end = backend.add_to_batch(
        batch_start,
        rus_circuit_list[-1],
        n_shots=n_shots,
        wasm_file_handler=wfh,
        batch_end=True,
    )
    batch_handles[rus_circuit_list[-1].name] = batch_end

    return batch_handles, experiment_cost


def rus_experiment(
    limit_start: int,
    limit_end: int,
    cond_execute: int,
    n_shots: int,
    max_batch_cost: float = 500,
    check_cost: bool = True,
    run_exp: bool = False,
):
    """Function for running a series of RUS experiments."""
    experiment_cost = []
    batch_handles = {}

    backend = QuantinuumBackend(device_name=machine)

    for limit in range(limit_start, limit_end + 1):

        circuit = build_rus_circuit(limit, cond_execute)
        compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

        if check_cost:
            circuit_cost = None
            while circuit_cost is None:
                circuit_cost = backend.cost(
                    compiled_circuit,
                    n_shots=n_shots,
                    syntax_checker=machine.replace("E", "SC"),
                )
            experiment_cost.append(circuit_cost)

        if run_exp:
            if limit == limit_start:
                batch_start = backend.start_batch(
                    max_batch_cost=max_batch_cost,
                    circuit=compiled_circuit,
                    n_shots=n_shots,
                    wasm_file_handler=wfh,
                )
                batch_handles[compiled_circuit.name] = batch_start
            elif limit == limit_end:
                batch_end = backend.add_to_batch(
                    batch_start,
                    compiled_circuit,
                    n_shots=n_shots,
                    wasm_file_handler=wfh,
                    batch_end=True,
                )
                batch_handles[compiled_circuit.name] = batch_end
            else:
                batch_mid = backend.add_to_batch(
                    batch_start,
                    compiled_circuit,
                    n_shots=n_shots,
                    wasm_file_handler=wfh,
                )
                batch_handles[compiled_circuit.name] = batch_mid

    return batch_handles, experiment_cost


# Set up the experiment parameters.


machine = "H1-1E"
n_shots = 10
cond_execute = 3
limit_start = 1
limit_end = 20
max_batch_cost = 500


# Check the cost in HQCs for running the experiment before running.


batch_handles, experiment_cost = run_rus_experiment(
    machine=machine,
    n_shots=n_shots,
    max_batch_cost=max_batch_cost,
    cond_execute=cond_execute,
    limit_start=limit_start,
    limit_end=limit_end,
)
print("Total Experiment Cost:", experiment_cost)


# The total cost of the experiment with 1,000 shots each is greater than a max batch cost, but submitting in batch will keep the up to the max batch cost (500 HQCs) run together.
#
# Submit the full set of experiments.


check_cost = False
run_exp = True
batch_handles, experiment_cost = rus_experiment(
    limit_start,
    limit_end,
    cond_execute,
    machine,
    n_shots,
    check_cost=check_cost,
    run_exp=run_exp,
)


# Check the status of the jobs.


status_list = [backend.circuit_status(h) for k, h in batch_handles.items()]
num_completed = len([x for x in status_list if x.status.name == "COMPLETED"])
print(f"{num_completed}/{len(status_list)} completed")


# Save results once completed, in case you need to come back to analysis later.


result = {}
result_json = {}

for key, handle in batch_handles.items():
    result[key] = backend.get_result(handle)
    result_json[key] = backend.get_result(handle).to_dict()

with open("RUS_experiments.json", "w", encoding="utf-8") as file:
    json.dump(result_json, file)


# Load saved results, if coming back to the experiment results.
from pytket.backends.backendresult import BackendResult

with open("RUS_experiments.json") as file:
    data = json.load(file)

result = {}
for key, handle in data.items():
    result[key] = BackendResult.from_dict(handle)


# Plot results.


import matplotlib.pyplot as plt

plt.figure()

# Plot limit vs. Fidelity (Count of 0%)
plt.errorbar(
    [k[0] for k, v in creg1_0_cnts.items()],
    [v[1] for k, v in creg1_0_cnts.items()],
    label="RUS",
    marker="o",
    alpha=1.0,
    color="red",
)

plt.ylabel("Logical fidelity")
plt.xlabel("limit")
plt.legend(loc="lower left")
plt.show()
