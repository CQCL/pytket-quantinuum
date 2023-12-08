# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # How to do Batch Submissions on H-Series Devices

# This notebook contains an example of how to perform batch submissions on H-Series emulators and quantum computers via `pytket`.

# The batch feature on H-Series backends gives users the ability to create "ad-hoc" reservations. Circuits submitted together in a batch will run at one time. The benefit to users is that once a batch hits the front of the queue, jobs in a batch will run uninterrupted until they are completed.

# Once a batch is submitted, jobs can continue to be added to the batch, ending either when the user signifies the end of a batch or after 1 minute of inactivity.

# Batches cannot exceed the maximum limit of 2,000 H-System Quantum Credits (HQCs) total. If the total HQCs for jobs in a batch hit this limit or a smaller limit set by the user, those jobs *will not be cancelled*. Instead, they will continue to run as regular jobs in the queue instead of as a batch.

# Currently only the quantum computer and emulator targets support the batching feature. Batching is not supported on the syntax checkers.

# For more information on using this feature in `pytket-quantinuum`, see [Batching](https://cqcl.github.io/pytket-quantinuum/api/index.html#batching).

# To start a batch, use the `start_batch` function, specifying the `max_batch_cost` in HQCs to enforce.

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.circuit.display import render_circuit_jupyter
from pytket.circuit import Circuit, fresh_symbol

machine = "H1-1E"
n_shots = 100
max_batch_cost = 100

# Set up Bell State
circuit = Circuit(2, name="Bell State")
circuit.H(0)
circuit.CX(0, 1)
circuit.measure_all()

backend = QuantinuumBackend(device_name=machine)

compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=0)
batch1 = backend.start_batch(
    max_batch_cost=max_batch_cost, circuit=compiled_circuit, n_shots=n_shots
)

# Additional jobs can be added to the batch using the `add_to_batch` function. The end of a batch can optionally be specified with the `batch_end` flag.

batch2 = backend.add_to_batch(batch1, compiled_circuit, n_shots=n_shots)
batch3 = backend.add_to_batch(batch1, compiled_circuit, n_shots=n_shots, batch_end=True)

# The status for the batch jobs can be checked once submitted.

handle_list = [batch1, batch2, batch3]
status_list = [backend.circuit_status(h) for h in handle_list]

status_list

# Results for batch submissions can be returned using `get_results` (note the plural).

results = backend.get_results(handle_list)
for result in results:
    print(result.get_counts())

# <div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>
