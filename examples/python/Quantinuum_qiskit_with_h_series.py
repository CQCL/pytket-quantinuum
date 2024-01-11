# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Using Qiskit with Quantinuum Devices

# Qiskit is a popular tool for quantum computing programmers. There are 2 ways to use Qiskit in conjunction with submitting to Quantinuum systems, which are outlined here.

# 1. [Use Qiskit in conjunction with Pytket and Pytket-Quantinuum](#qiskit-pytket)
# 2. [Use the Qiskit Quantinuum Provider from the Qiskit Community](#qiskit-quantinuum-provider)

# ## Use Qiskit in conjunction with Pytket and Pytket-Quantinuum <a class="anchor" id="qiskit-pytket"></a>

# Here we highlight the compatability of `qiskit` with `pytket` and `pytket-quantinuum` for submitting to Quantinuum devices.

# **Note:** Not all capabilities for Quantinuum devices available via `pytket` are guaranteed to be available in `qiskit`. Some use cases may require working directly in pytket.*

# Similar `pytket` workflows for submitting to Quantinuum devices exist for other programming interfaces such as `cirq` or Q#. For more information, see the full list of [pytket-extensions](https://tket.quantinuum.com/api-docs/extensions.html).

# Running this notebook requires the `pytket-qiskit` extension. Run `pip install pytket-qiskit` in your python environment before running this notebook. This will also install Qiskit if it isn't installed already.

# ### Circuit Preparation

# When working with `qiskit`, quantum circuits are created using Qiskit's `QuantumCircuit` object.

from qiskit import QuantumCircuit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.circuit.display import render_circuit_jupyter

n_qubits = 2
circuit = QuantumCircuit(n_qubits, n_qubits, name="Bell Test")
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

circuit.draw("mpl")

# ### Convert to pytket

# To submit to Quantinuum devices with `pytket-quantinuum`, the Qiskit `QuantumCircuit` object needs to be converted to a Pytket `Circuit` object. This is done via the `qiskit_to_tk` function. Quantum circuits can be converted between Qiskit and Pytket options via the `qiskit_to_tk` and `tk_to_qiskit` functions.

tket_circuit = qiskit_to_tk(circuit)

render_circuit_jupyter(tket_circuit)

# ### Submit to Quantinuum Devices

# Now you're ready to submit to Quantinuum devices! Once you have a Pytket `Circuit` object, you can use the `pytket-quantinuum` interface for submitting to Quantinuum devices. An example of how to do this is found in the notebook `02 - How to Submit Quantum Circuits to H-Series Backends.ipynb` on the User Portal or [Quantinuum_circuit_submissions.ipynb](https://github.com/CQCL/pytket-quantinuum/blob/develop/examples/Quantinuum_circuit_submissions.ipynb) in the `examples` folder on [pytket-quantinuum](https://github.com/CQCL/pytket-quantinuum).

# ## Use the Qiskit Quantinuum Provider from the Qiskit Community <a class="anchor" id="qiskit-quantinuum-provider"></a>

# Another option for using Qiskit with Quantinuum systems is to use the Qiskit Quantinuum provider available on the Qiskit community: [qiskit-quantinuum-provider](https://github.com/qiskit-community/qiskit-quantinuum-provider)

# *Note that not all capabilities for Quantinuum devices available via `pytket` are guaranteed to be available in `qiskit`. Some use cases may require working directly in pytket.*

# An example of how to submit to Quantinuum systems is provided in the [examples](https://github.com/qiskit-community/qiskit-quantinuum-provider/tree/master/examples) folder within the Github repository.

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
