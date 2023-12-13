# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Discarding leaky results: Automatic leakage error detection with `QuantinuumBackend`

# Quantum computers are known to be *noisy*, with a high chance of errors occurring when executing a sequence of operations. These errors can come from a variety of sources and are typically hard to mitigate. Investigating the source of errors, how they manifest at the quantum circuit level, and how to mitigate them, is a wide area of research.

# A regular experiment using the `QuantinuumBackend` works as follows: circuits are defined using `pytket`, compiled to the device constraints using `QuantinuumBackend.get_compiled_circuit` and sent to the device provider using `QuantinuumBackend.process_circuits`; results are retrieved using `QuantinuumBackend.get_results`. Results are received in a `BackendResult` object and represented as a series of "shots", a binary string corresponding to the measurement results for each `Qubit` after each execution of the circuit on the device. This set of shots gives a probability distribution over the set of basis states spanned by the device qubits. Broadly, the closer the probability distribution represented by the set of shots matches the ideal probability distribution, the more accurate the computation can be considered (note we are considering noise here, but the number of shots taken also affects this).

# One source of error in computation is from "leakage". During execution of hardware level two-qubit gates, there is a small probability for the qubit to experience leakage to electronic states outside the qubit subspace. When a qubit has leaked, none of the remaining gates in the circuit acting on the qubit have any effect, but a measurement on that qubit will falsely result in a '1'. For more information we refer to [Eliminating Leakage Errors in Hyperfine Qubits](https://arxiv.org/abs/1912.13131) by D. Hayes, D. Stack, B. Bjork, A. C. Potter, C. H. Baldwin and R. P. Stutz.

# Using a special circuit "gadget" and an ancilla qubit, it's possible to detect leakage errors at the quantum circuit level and so discard erroneous shots. In this notebook we will show how to automatically run experiments with leakage detection added using `QuantinuumBackend.process_circuits` and how to process `BackendResult` returned from `QuantinuumBackend` to remove leaky results. We will also show how this scheme can improve results using a repeated gate benchmarking experiment.

# The leakage detection circuit, or "gadget", looks like so:

from pytket.extensions.quantinuum.backends.leakage_gadget import (
    get_leakage_gadget_circuit,
)
from pytket.circuit.display import render_circuit_jupyter
from pytket import Qubit, Bit

render_circuit_jupyter(
    get_leakage_gadget_circuit(
        Qubit("experiment_qubit", 0),
        Qubit("leakage_detection_qubit", 0),
        Bit("leakage_detection_bit", 0),
    )
)

# The action of this Circuit is such that if leakage hasn't occurred, then the `experiment_qubit` state is unchanged and the `leakage_detection_qubit` is in state "0", while if leakage has occurred then the `leakage_detection_qubit` will be in state "1" and so can easily be detected.

# The `pytket-quantinuum` package has methods for automatically modifying a `pytket` `Circuit` to add leakage detection gadgets just before any end of circuit measurement gates.

# Let's create a basic Bell pair circuit and then modify it.

from pytket.extensions.quantinuum.backends.leakage_gadget import get_detection_circuit
from pytket import Circuit

bell_pair_circuit = Circuit(2).H(0).CX(0, 1).measure_all()
bell_pair_leakage_detection_circuit = get_detection_circuit(
    circuit=bell_pair_circuit, n_device_qubits=4
)

render_circuit_jupyter(bell_pair_leakage_detection_circuit)

# We can see that for each of the Bell pair qubits a leakage detection gadget has been appended before the final measurement.

# The parameter `n_device_qubits` tells `get_detection_circuit` how many qubits the device has. In this case we stated there were 4 device qubits while the Bell pair circuit had 2 qubits, meaning a separate qubit was used for each leakage detection gadget. However, if there are too few device qubits then `get_detection_circuit` will reuse ancilla qubits to do multiple leakage detections, assigning the results to different `Bit`. We can see this by setting `n_device_qubits = 3`.

# render_circuit_jupyter(get_detection_circuit(circuit=bell_pair_circuit, n_device_qubits=3))

# The `QuantinuumBackend.process_circuits` method takes an optional `kwarg` `leakage_detection` which when `True`, will pass every circuit through `get_detection_circuit` before passing it to the hardware. This means all circuits will be executed with leakage detection added.

# When passing a `ResultHandle` to `QuantinuumBackend.get_result`,  the retrieved `BackendResult` will include measurement results from the leakage detection qubits.

from pytket.extensions.quantinuum import QuantinuumBackend

circuit = Circuit(2, 2).H(0).CX(0, 1).measure_all()
# Note for demonstration purposes this is not calling the real hardware.
backend = QuantinuumBackend(device_name="H1-1E")
handle = backend.process_circuit(circuit, n_shots=10000, leakage_detection=True)
result = backend.get_result(handle)

# We can see in the returned results that there are additional `Bit` for detecting leakage. </br>

# By passing an ordering of `Bit` to `result.get_counts()`, we ensure that the final two entries in each bitstring correspond to the measurement results for the leakage detection Qubit. </br>

# If each of these entries is a "1" then leakage has occurred and these counts should be removed.

print("Bit with measurement results:", list(result.c_bits.keys()))
for bitstring, count in result.get_counts(
    [Bit(0), Bit(1), Bit("leakage_detection_bit", 0), Bit("leakage_detection_bit", 1)]
).items():
    print(bitstring, count)

# We can now use the method `prune_shots_detected_as_leaky` to modify this `BackendResult` object, removing counts corresponding to leaked results.

from pytket.extensions.quantinuum import prune_shots_detected_as_leaky

pruned_result = prune_shots_detected_as_leaky(result)
print("Bit with measurement results:", list(pruned_result.c_bits.keys()))
for bitstring, count in pruned_result.get_counts([Bit(0), Bit(1)]).items():
    print(bitstring, count)

# ## Benchmarking circuit improvement with leakage detection

# In this section we will demonstrate how using leakage detection can improve the results of a circuit. Since the probability of a qubit having leaked during a circuit increases with the number of two-qubit gates that the qubit participates in, we use a deep circuit in order to measure a statistically significant improvement. The following circuit repeats the native two-qubit gate many times and is such that the final state is ideally $| 00\rangle$.

circuit = Circuit(2, 2)
for _ in range(200):
    circuit.ZZMax(0, 1)
    circuit.add_barrier([0, 1])
circuit.measure_all()

# We next run the circuit through the Quantinuum H1-1 emulator, which realistically models leakage during the two-qubit gates. We create two circuit handles, one without and the other with leakage detection.

backend = QuantinuumBackend(device_name="H1-1E")
compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)
handle_no_leakage = backend.process_circuit(
    compiled_circuit, n_shots=10000, leakage_detection=False
)
handle_leakage = backend.process_circuit(
    compiled_circuit, n_shots=10000, leakage_detection=True
)

# When submitting circuits to the Quantinuum hardware emulator, it is a good idea to check the status of the circuits.

for handle in [handle_no_leakage, handle_leakage]:
    print(backend.circuit_status(handle).status)

# We now get the results from the circuits, and compare the success probabilities.

import numpy as np

counts_no_leakage = backend.get_result(handle_no_leakage).get_counts()
counts_leakage = prune_shots_detected_as_leaky(
    backend.get_result(handle_leakage)
).get_counts()

print("Counts without leakage detection:", counts_no_leakage)
print("Counts with leakage detection:", counts_leakage)

n_shots_no_leakage = sum(counts_no_leakage.values())
n_shots_leakage = sum(counts_leakage.values())

prob_no_leakage = counts_no_leakage.get((0, 0)) / n_shots_no_leakage
prob_leakage = counts_leakage.get((0, 0)) / n_shots_leakage

std_no_leakage = np.sqrt(prob_no_leakage * (1 - prob_no_leakage) / n_shots_no_leakage)
std_leakage = np.sqrt(prob_leakage * (1 - prob_leakage) / n_shots_leakage)

print(
    f"Success probability without leakage detection: {round(prob_no_leakage,4)} +/- {round(std_no_leakage,4)}"
)
print(
    f"Success probability with leakage detection:    {round(prob_leakage,4)} +/- {round(std_leakage,4)}"
)

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
