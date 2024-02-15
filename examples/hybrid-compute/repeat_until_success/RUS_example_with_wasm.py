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

machine = "H1-1SC"
backend = QuantinuumBackend(device_name=machine)
backend.login()

# #### Set up Circuit
#
# Set up the RUS circuit. This involves 2 steps:
# 1. Set up Wasm File handler. This checks the Wasm file and can list the available functions within the file.
# 2. Set up the pytket circuit, which calls the Wasm file handler.


import pathlib
from pytket import wasm

rus_dir = pathlib.Path().cwd().joinpath("RUS_WASM")
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

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)

n_shots = 1000
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")
handle = backend.process_circuit(
    compiled_circuit, n_shots=n_shots, wasm_file_handler=wfh
)
result = backend.get_result(handle)
print(result)

import json

with open(compiled_circuit.name + ".json", "w", encoding="utf-8") as file:
    json.dump(result.to_dict(), file)


from pytket.unit_id import Bit
from pytket.backends.backendresult import BackendResult


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


result1 = {}
result1[compiled_circuit.name] = result
creg1_0_cnts, creg1_1_cnts = get_creg1_counts_from_rus_result(result1)
print(creg1_0_cnts, creg1_1_cnts)
