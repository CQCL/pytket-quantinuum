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


circuit = build_rus_circuit(5, 3)

# compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)
#
# n_shots = 1000
# backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")
#
#
# handle = backend.process_circuit(
#    compiled_circuit, n_shots=n_shots, wasm_file_handler=wfh
# )
#
# result = backend.get_result(handle)
#
# print(result)

import json

# with open(compiled_circuit.name + ".json", "w", encoding="utf-8") as file:
#    json.dump(result.to_dict(), file)
#
from pytket.unit_id import Bit
from pytket.backends import BackendResult


def get_creg1_counts_from_rus_result(
    result_dict: dict,
) -> tuple[dict[str, BackendResult]]:
    """Analyze results of RUS experiment."""
    creg1_0_cnts = {}
    creg1_1_cnts = {}

    for key, res in result_dict.items():
        creg1_counts = res.get_counts(cbits=[Bit("creg1", 0)])
        creg1_0_cnts[key] = creg1_counts[(0,)]
        creg1_1_cnts[key] = creg1_counts[(1,)]

    return creg1_0_cnts, creg1_1_cnts


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


# result1 = {}
# result1[compiled_circuit.name] = result
# creg1_0_cnts, creg1_1_cnts = get_creg1_counts_from_rus_result(result1)


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


machine = "H1-1E"
n_shots = 10
cond_execute = 3
limit_start = 1
limit_end = 20
max_batch_cost = 500


batch_handles, experiment_cost = run_rus_experiment(
    machine=machine,
    n_shots=n_shots,
    max_batch_cost=max_batch_cost,
    cond_execute=cond_execute,
    limit_start=limit_start,
    limit_end=limit_end,
)
print("Total Experiment Cost:", experiment_cost)

status_list = [backend.circuit_status(h) for k, h in batch_handles.items()]
num_completed = len([x for x in status_list if x.status.name == "COMPLETED"])
print(f"{num_completed}/{len(status_list)} completed")

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

creg1_0_cnts, creg1_0_cnts = get_creg1_counts_from_rus_result(result)


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
