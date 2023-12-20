# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Mid-Circuit Measurement

# This notebook performs a repetition code calculation with the following H-Series features:

# * Qubit Reuse via Mid-circuit measurements with reset
# * Classicaly-conditioned operations

# The notebook performs a D=3, T=1 repetition code. Three physical qubits are used to encode one logical qubit. The physical qubit register is initialised in the $|000\rangle$ state encoding the logical $|0\rangle$ state.

# One ancilla qubit is used to perform two syndrome measurements:

# 1. $\hat{Z}_{q[0]} \hat{Z}_{q[1]} \hat{I}_{q[2]}$
# 2. $\hat{I}_{q[0]} \hat{Z}_{q[1]} \hat{Z}_{q[2]}$

# Subsequently, classically-conditioned operations are used to correct any errors on the physical qubits using the syndrome measurement results. Finally, direct measurements on the physical qubits are performed to verify the final state of the logical qubit is $|0\rangle$.

# ## Syndrome Measurement Circuit Primitive

# In the code cell below, a circuit primitive is defined to detect errors on two physical qubits with one ancilla qubit.

from pytket.circuit import Circuit, OpType, CircBox
from pytket.circuit.display import render_circuit_jupyter


def syndrome_extraction():
    circuit = Circuit(3, 1)
    circuit.CX(1, 0)
    circuit.CX(2, 0)
    circuit.Measure(0, 0)
    circuit.add_gate(OpType.Reset, [0])
    return CircBox(circuit)


syndrome_box = syndrome_extraction()
render_circuit_jupyter(syndrome_box.get_circuit())

# ## Repetition Code Circuit

# Initially, a `pytket.circuit.Circuit` is instantiated with three physical qubits (`data` register), one ancilla qubits (`ancilla` register). Additionally, two classical registers are added: the first to store output from syndrome measurements (`syndrome` register); and the second (`output` register) to store output from direct measurement on phyiscal qubits.

# The use of mid-circuit measurement is straightforward. Note the use of `measure` and `reset` on the ancilla qubits. This example also utilizes conditional logic available with Quantinuum devices as well as Registers and IDs available in `pytket`. See [Classical and conditional operations](https://tket.quantinuum.com/user-manual/manual_circuit.html#classical-and-conditional-operations) and [Registers and IDs](https://tket.quantinuum.com/user-manual/manual_circuit.html#registers-and-ids) for additional examples.

# The circuit is named "Repetition Code". This name is used by the Job submitted to H-series later in this notebook.

from pytket.circuit import Circuit

# Set up circuit object
circuit = Circuit(name="Repetition Code")

# Reserve registries

# Add qubit register, the data qubits
data = circuit.add_q_register("data", 3)

# Add qubit register, the ancilla qubit
ancilla = circuit.add_q_register("ancilla", 1)

# Add classical registers for the syndromes
syndrome = circuit.add_c_register("syndrome", 2)

# Add classical registers for the output
output = circuit.add_c_register("output", 3)

# The syndrome measurement primitive, defined above, is added twice as `pytket.circuit.CircBox`. The first measures $\hat{Z}_{q[0]} \hat{Z}_{q[1]} \hat{I}_{q[2]}$ and the second measures $\hat{I}_{q[0]} \hat{Z}_{q[1]} \hat{Z}_{q[2]}$. This is one round of syndrome measurements. The  `CircBox` instances are decomposed with `pytket.passes.DecomposeBoxes`.

from pytket.passes import DecomposeBoxes

# Syndrome Extraction 1: ZZI
circuit.add_circbox(syndrome_box, [ancilla[0], data[0], data[1], syndrome[0]])

# Syndrome Extraction 2: IZZ
circuit.add_circbox(syndrome_box, [ancilla[0], data[1], data[2], syndrome[1]])
DecomposeBoxes().apply(circuit)

# In the cell below, classically-conditioned operations (`pytket.circuit.OpType.X`) are performed using `pytket.circuit.logic_exp.reg_eq`. The function, `reg_eq`, checks if the measurement output stored in the classical register is equivalent to a particular value. If the equiavlence check is `True`, the desired operation is applied to the specified qubit.

# The `X` operation is applied to qubit `data[0]`. The reg_ex checks if the classical output is 01 (little endian - syndrome[0] = 1 and syndrome[1] = 0).

from pytket.circuit.logic_exp import reg_eq

circuit.X(data[0], condition=reg_eq(syndrome, 1))

# The `X` operation is applied to qubit `data[2]`. The reg_ex checks if the classical output is 10 (syndrome[0] = 0 and syndrome[1] = 1). If there is no error from the first syndrome measurement (syndrome[0] = 0), but error from the second syndrome measurement (syndrome[1] = 1), then there is a bitflip on the qubit `data[2]`.

# if(syndromes==2) -> 01 -> check 1 bad -> X on qubit 2
circuit.X(data[2], condition=reg_eq(syndrome, 2))

# The `X` operation is applied to qubit `data[1]`. The reg_ex checks if the classical output is 11 (syndrome[0] = 1 and syndrome[1] = 1). If there is error from the first syndrome measurement (syndrome[0] = 1) and error from the second syndrome measurement (syndrome[1] = 1), then there is a bitflip on the qubit `data[1]`.

# # if(syndromes==3) -> 11 -> check 1 and 2 bad -> X on qubit 1
circuit.X(data[1], condition=reg_eq(syndrome, 3))

# Finally, measurement gates are added to the `data` qubit register.

# Measure out data qubits
circuit.Measure(data[0], output[0])
circuit.Measure(data[1], output[1])
circuit.Measure(data[2], output[2])

# The display tool in pytket is used to visualise the circuit in jupyter.

from pytket.circuit.display import render_circuit_jupyter

render_circuit_jupyter(circuit)

## Select Device

# Login to the Quantinuum API using your credentials and check the device status.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

print(machine, "status:", backend.device_state(device_name=machine))

# ### Circuit Compilation

# `pytket` includes many features for optimizing circuits. This includes reducing the number of gates where possible and resynthesizing circuits for a quantum computer's native gate set. See the `pytket` [User Manual](https://tket.quantinuum.com/user-manual/) for more information on all the options that are available.

# Here the circuit is compiled with `get_compiled_circuit`, which includes optimizing the gates and resynthesizing the circuit to Quantinuum's native gate set. The `optimisation_level` sets the level of optimisation to perform during compilation, check [Default Compilation](https://cqcl.github.io/pytket-quantinuum/api/index.html#default-compilation) in the pytket-quantinuum documentation for more details.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=2)

render_circuit_jupyter(compiled_circuit)

# ## Submit and Run the Circuit

n_shots = 100
h1_cost = backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")
print(f"Cost: {h1_cost} HQC")

handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
print(handle)

status = backend.circuit_status(handle)
print(status)

import json

result = backend.get_result(handle)

with open("pytket_mcmr_example.json", "w") as file:
    json.dump(result.to_dict(), file)

# ## Analyze Results

# We will now take the raw results and apply a majority vote to determine how many times we got 0 vs 1.

# First, define a majority vote function.


def majority(result):
    """Returns whether the output should be considered a 0 or 1."""
    if result.count(0) > result.count(1):
        return 0
    elif result.count(0) < result.count(1):
        return 1
    else:
        raise Exception("count(0) should not equal count(1)")


# Now process the output:

result_output_cnts = result.get_counts([output[i] for i in range(output.size)])

result_output_cnts

# Here, determine how many times 0 vs 1 was observed using the majority vote function.

zeros = 0  # Counts the shots with majority zeros
ones = 0  # Counts the shots with majority ones

for out in result_output_cnts:
    m = majority(out)

    if m == 0:
        zeros += result_output_cnts[out]
    else:
        ones += result_output_cnts[out]

# A logical zero was initialized, so our error rate should be number of ones / total number of shots: `ones/shots`

p = ones / n_shots
print(f"The error-rate is: p = {p}")

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
