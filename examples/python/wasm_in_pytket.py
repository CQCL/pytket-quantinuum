# # Wasm Calls with pytket

# The Wasm module in pytket allows you to add external classical functions from compiled web assembly (Wasm) to a quantum circuit. To begin, you need a compiled Wasm file that contains functions you'd like to call from your quantum circuit.

# ## Set up `WasmFileHandler`

# Once you have a compiled Wasm file, you will create a `WasmFileHandler` object to call and use the compiled Wasm within your quantum circuit. The `WasmFileHandler` knows all available functions and the corresponding signatures within the Wasm file.

from pytket import wasm, Circuit, Bit

wfh = wasm.WasmFileHandler("testfile.wasm")
print("wasm file uid:")
print(wfh)

# If you are not sure about the signatures of the functions of your file, you can get a list of them from the `WasmFileHandler` as shown below. The parameters and result types of the supported functions must be `i32`. All functions that contain other types will be listed when printing the `WasmFileHandler` as well, but you will not be able to add them to a quantum circuit.

print("wasm file repr:")
print(repr(wfh))

# ## Add classical function calls to your quantum circuit

# Next, we add the classical function calls to our quantum circuit.

# First we use the [add_wasm](https://cqcl.github.io/tket/pytket/api/circuit_class.html#pytket.circuit.Circuit.add_wasm) function and add the function `add_one`, defined in the WASM file. The first parameter will be read from `Bit(0)` and the result written to `Bit(1)`. The length of the two lists giving the number of bits is the number of parameters and the number of results. For more information on the `add_wasm` function, see [add_wasm](https://cqcl.github.io/tket/pytket/api/circuit_class.html#pytket.circuit.Circuit.add_wasm).

c = Circuit(0, 8)

c.add_wasm(
    funcname="add_one",  # Function in the Wasm file
    filehandler=wfh,  # Wasm file handler
    list_i=[1],  # Number of bits in the input variables in i32 format
    list_o=[1],  # Number of bits in the output variables in i32 format
    args=[Bit(0), Bit(1)],  # List of circuit bits where the wasm op will be added to
)

# If you have more than one bit per parameter, you can add as shown below. This will add the function `add_one` to read from `Bit(0)` and `Bit(1)` for the first parameter and write the result to `Bit(2)`, `Bit(3)` and `Bit(4)`.

c.add_wasm("add_one", wfh, [2], [3], [Bit(0), Bit(1), Bit(2), Bit(3), Bit(4)])

# Functions with multiple parameters can be done in the same way.

c.add_wasm(
    "multi",
    wfh,
    [2, 1],
    [3],
    [Bit(0), Bit(1), Bit(5), Bit(2), Bit(3), Bit(4)],
)

# If you want to add two parameters with the same bits, that is fine too.

c.add_wasm(
    "multi", wfh, [2, 2], [3], [Bit(0), Bit(1), Bit(0), Bit(1), Bit(2), Bit(3), Bit(4)]
)

# If you are working with registers in your circuit as a means to organize the classical bits, you can add Wasm to your circuit using registers for each parameter and result. For more information on the `add_wasm_to_reg` function, see [add_wasm_to_reg](https://cqcl.github.io/tket/pytket/api/circuit_class.html#pytket.circuit.Circuit.add_wasm_to_reg).

# Add registers to circuit
c0 = c.add_c_register("c0", 3)
c1 = c.add_c_register("c1", 4)
c2 = c.add_c_register("c2", 5)

c.add_wasm_to_reg("multi", wfh, [c0, c1], [c2])
c.add_wasm_to_reg("add_one", wfh, [c2], [c2])

# Your Wasm file might have some global data. To make sure this data is not affected by function calls in the wrong order, pytket will make sure that the order of the Wasm calls within a circuit is not restructured. For this purpose, pytket will add all wasm operations to a `wasm_wire` by default. If you are not worried about a possible restructure of the Wasm calls in your circuit, you have the option to not add the `wasm_wire` to your Wasm operations. If you only want to stop some special reordering for some of your wasm operations, you can add some the wasm operations to multiple `wasm_wire` to enable the restructuring in the intended way. Even if there are no `wasm_wire` given, pytket will only restructure the wasm operations if there are no dependencies in parameters or the results.

# Here you can see that all operations we have created above are conected to the default `wasm_wire`:

for gate in c:
    print(gate)

# We will now create a new circuit and add four operations. The two `add_one` operations should be allowed to commute, but we want to make sure that `multi` is executed after the two other functions. The last `add_two` operation can commute with all others.

c = Circuit(0, 5)

c.add_wasm("add_one", wfh, [1], [1], [Bit(0), Bit(0)], [0])
c.add_wasm("add_one", wfh, [1], [1], [Bit(1), Bit(1)], [1])
c.add_wasm("multi", wfh, [1, 1], [1], [Bit(2), Bit(3), Bit(2)], [0, 1])
c.add_wasm("add_two", wfh, [1], [1], [Bit(4), Bit(4)], [])

for gate in c:
    print(gate)

# One helpful feature is to plot the DAG of the circuit to get an overview of the different components of the circuit.

# from pytket.utils import Graph

# g = Graph(c)
# g.view_DAG()

# ## Send Wasm to the Backend

# In the last step we want to send the circuit with wasm to a backend. First we create the backend. For this step you will need Quantinuum credentials.

from pytket.extensions.quantinuum import QuantinuumBackend

machine = "H1-1E"
b = QuantinuumBackend(device_name=machine)
b.login()

# When submitting the circuit to run in `process_circuits`, the `WasmFileHandler` is added as an input.

c = Circuit(1, name="Test Wasm")
a = c.add_c_register("a", 8)
c.add_wasm_to_reg("add_one", wfh, [a], [a])
c = b.get_compiled_circuit(c)
h = b.process_circuits([c], n_shots=10, wasm_file_handler=wfh)[0]

status = b.circuit_status(h)
print(status)

# result = b.get_result(h)
# print(result)

# for shot in result.get_shots():
#     print(shot)

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
