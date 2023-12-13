# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Conditional Gates

# Quantinuum H-Series hardware and pytket support conditional gating. This may be for implementing error correction or reducing noise. This capability is well-supported by Quantinuum hardware, which supports mid-circuit measurement and reset and qubit reuse. See [Conditional Gates](https://tket.quantinuum.com/user-manual/manual_circuit.html#classical-and-conditional-operations) for more information on pytket's implementation. The following example demonstrates a quantum teleportation protocol.

from pytket.circuit import Circuit, if_bit
from pytket.circuit.display import render_circuit_jupyter

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

# We can utilise pytket's [Assertion](https://tket.quantinuum.com/user-manual/manual_assertion.html) feature to verify the successful teleportation of the state $| - \rangle$.

from pytket.circuit import ProjectorAssertionBox
import numpy as np

# |-><-|
proj = np.array([[0.5, -0.5], [-0.5, 0.5]])
circ.add_assertion(ProjectorAssertionBox(proj), [qreg[2]], name="debug")

render_circuit_jupyter(circ)

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
n_shots = 100
backend = QuantinuumBackend(device_name=machine)
compiled_circuit = backend.get_compiled_circuit(circ)
backend.cost(compiled_circuit, n_shots=n_shots, syntax_checker="H1-1SC")
handle = backend.process_circuit(compiled_circuit, n_shots=n_shots)
status = backend.circuit_status(handle)
status

result = backend.get_result(handle)

# The `get_debug_info` function returns the success rate of the state assertion averaged across shots. Note that the failed shots are caused by the simulated device errors

result.get_debug_info()

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
