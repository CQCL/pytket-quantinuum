# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Compiling for Quantinuum Hardware without Querying Quantinuum API

# This notebook contains an example of how to investigate circuits compiled for Quantinuum hardware without logging in or submitting to Quantinuum hardware. This may be useful if it is desired to explore circuit compilation in depth before submitting.

# ## Circuit Preparation <a class="anchor" id="circuit-preparation"></a>

# Create your circuit.

from pytket.circuit import Circuit, OpType
from pytket.circuit.display import render_circuit_jupyter

circuit = Circuit(2, name="Bell Test")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()

render_circuit_jupyter(circuit)

# ## Set up Backend

# Set up a `QuantinuumBackend` object. The difference is the `machine_debug` option uses the default `pytket-quantinuum` options such as pytket's version of the Quantinuum native gate set rather than querying the Quantinuum API for this information.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
backend = QuantinuumBackend(device_name=machine, machine_debug=True)

# ## Investigate Native Gate Set

# Users can view the hard-coded native gate set for the Quantinuum backend using the following command.

import pytket.extensions.quantinuum.backends.quantinuum as qtm

print(qtm._GATE_SET)

# It's possible that the hardcoded verion is not up to date with the latest native gate set as described in the [System Model H1 Product Data Sheet](https://www.quantinuum.com/hardware/h1). In this case, the Rzz gate, which is the `ZZPhase` gate in pytket, is missing. This can be added by running the following command.

qtm._GATE_SET.add(OpType.ZZPhase)

# ### Circuit Compilation <a class="anchor" id="circuit-compilation"></a>

# Circuits can now be compiled with the `get_compiled_circuit` function without querying the Quantinuum API.

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=2)

render_circuit_jupyter(compiled_circuit)

# <div align="center"> &copy; 2022 by Quantinuum. All Rights Reserved. </div>
