# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # How to Submit Parametrized Circuits

# Parametrized circuits are common in variational algorithms. Pytket supports parameters within circuits via symbols. For more information, see [Symbolic Circuits](https://cqcl.github.io/pytket/manual/manual_circuit.html?highlight=paramet#symbolic-circuits) and [Symbolic Compilation](https://tket.quantinuum.com/user-manual/manual_compiler.html#compiling-symbolic-circuits).

from pytket.circuit import fresh_symbol
from pytket.circuit import Circuit
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine)
backend.login()

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
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")
handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)

status = backend.circuit_status(handle)
print(status)

result = backend.get_result(handle)
print(result)

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
