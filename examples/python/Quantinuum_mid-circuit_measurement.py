# # Mid-Circuit Measurement

# This notebook contains an example using mid-circuit measurement using the Quantinuum machines.

# ## Repetition Code Circuit

# The use of mid-circuit measurement is straightforward, note the use of `measure` and `reset` on the ancilla qubits. This example also utlizes conditional logic available with Quantinuum devices as well as Registers and IDs available in `pytket`. See [Conditional Gates](https://cqcl.github.io/pytket/manual/manual_circuit.html#classical-and-conditional-operations) and [Registers and IDs](https://cqcl.github.io/pytket/manual/manual_circuit.html#registers-and-ids) for additional examples.

from pytket.circuit import Circuit, Qubit, Bit, OpType, reg_eq
from pytket.circuit.display import render_circuit_jupyter

# Set up Repetition Code with mid-circuit measurement and corrections

# 2    1    0 = data: data qubits
# *----*----*
#   ZZ   ZZ
#   1    0    = syndromes
#   0    0    = ancillas

# Set up circuit object
circuit = Circuit(name="Repetition Code")

# Reserve registries

# Add qubit register, the data qubits
data = circuit.add_q_register("data", 3)

# Add qubit register, the ancilla qubit
ancilla = circuit.add_q_register("anc", 1)

# Add classical registers for the syndromes
syndrome = circuit.add_c_register("syndrome", 2)

# Add classical registers for the output
output = circuit.add_c_register("output", 3)

# Prepare the logical state
# Qubits always start in |0> and logical |0> == |000>.
# So we already start in logical |0>.

# Syndrome Extraction
circuit.add_gate(OpType.Reset, ancilla)
circuit.CX(data[0], ancilla[0])
circuit.CX(data[1], ancilla[0])
circuit.Measure(ancilla[0], syndrome[0])

# Syndrome Extraction
circuit.add_gate(OpType.Reset, ancilla)
circuit.CX(data[1], ancilla[0])
circuit.CX(data[2], ancilla[0])
circuit.Measure(ancilla[0], syndrome[1])

# Correction
# # if(syndromes==1) -> 01 -> check 0 bad -> X on qubit 0
circuit.X(data[0], condition=reg_eq(syndrome, 1))

# # if(syndromes==2) -> 10 -> check 1 bad -> X on qubit 2
circuit.X(data[2], condition=reg_eq(syndrome, 2))

# # if(syndromes==3) -> 11 -> check 1 and 2 bad -> X on qubit 1
circuit.X(data[1], condition=reg_eq(syndrome, 3))

# Measure out data qubits
circuit.Measure(data[0], output[0])
circuit.Measure(data[1], output[1])
circuit.Measure(data[2], output[2])

render_circuit_jupyter(circuit)

# ## Select Device

# Login to the Quantinuum API using your credentials and check the device status.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = 'H1-1E'

backend = QuantinuumBackend(device_name=machine)

backend.login()

print(machine, "status:", backend.device_state(device_name=machine))

# ### Circuit Compilation

# `pytket` includes many features for optimizing circuits. This includes reducing the number of gates where possible and resynthesizing circuits for a quantum computer's native gate set. See the `pytket` [User Manual](https://cqcl.github.io/pytket/manual/index.html) for more information on all the options that are available.

# Here the circuit is compiled with `get_compiled_circuit`, which includes optimizing the gates and resynthesizing the circuit to Quantinuum's native gate set. The `optimisation_level` sets the level of optimisation to perform during compilation, check pytket documentation for more details.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=1)

render_circuit_jupyter(compiled_circuit)

# ## Submit and Run the Circuit

n_shots = 100
print("Cost in HQCs:", backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker='H1-1SC'))

handle = backend.process_circuit(compiled_circuit, 
                                 n_shots=n_shots)
print(handle)

status = backend.circuit_status(handle)
print(status)

import json 

result = backend.get_result(handle)

with open('pytket_mcmr_example.json', 'w') as file:
    json.dump(result.to_dict(), file)

# ## Analyze Results

# We will now take the raw results and apply a majority vote to determine how many times we got 0 vs 1.

# First, define a majority vote function.

def majority(result):
    """ Returns whether the output should be considered a 0 or 1. """
    if result.count(0) > result.count(1):
        return 0
    elif result.count(0) < result.count(1):
        return 1
    else:
        raise Exception('count(0) should not equal count(1)')

# Now process the output:

result_output_cnts = result.get_counts([output[i] for i in range(output.size)])

result_output_cnts

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
print(f'The error-rate is: p = {p}')

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
