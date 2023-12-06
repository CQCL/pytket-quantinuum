# <div style="text-align: center;">
# <img src="https://assets-global.website-files.com/62b9d45fb3f64842a96c9686/62d84db4aeb2f6552f3a2f78_Quantinuum%20Logo__horizontal%20blue.svg" width="200" height="200" /></div>

# # Conditional Execution on H-Series

# Quantinuum H-Series systems are capable of executing conditional workflows. This notebook presents how to do this in pytket. For more information, see [Conditional Execution](https://tket.quantinuum.com/examples/conditional_gate_example.html?highlight=qasm) and [Classical and conditional operations](https://cqcl.github.io/pytket/manual/manual_circuit.html#classical-and-conditional-operations).

# ## Classical assignment of registers or bits

# The value of classical registers and bits be assigned as illustrated below. Note that you can assign classical values at any point as well as re-assign values.

from pytket.circuit.display import render_circuit_jupyter
from pytket.circuit import (
    Circuit,
    BitRegister,
    if_bit,
    if_not_bit,
    reg_eq,
    reg_geq,
    reg_gt,
    reg_leq,
    reg_lt,
    reg_neq,
)

circuit = Circuit(name="Conditional Example")

qreg = circuit.add_q_register("q", 1)
reg_a = circuit.add_c_register("a", 10)
reg_b = circuit.add_c_register("b", 10)
reg_c = circuit.add_c_register("c", 10)

circuit.add_c_setbits([1], [reg_a[0]])  # a[0] = 1
circuit.add_c_setreg(2, reg_a)  # a = 2
circuit.add_c_setreg(3, reg_b)  # b = 3

# ## Binary operators

# Bitwise binary operators are avilable and can be applied to entire classical registers or bits. This allows one to update values based on mid-circuit measurement results as well as allow more advanced classical register comparisons.

circuit.add_classicalexpbox_register(reg_a ^ reg_b, reg_c)  # c = a ^ b
circuit.add_classicalexpbox_bit(reg_a[0] ^ reg_b[0], [reg_c[0]])  # c[0] = a[0] & b[0]
circuit.add_classicalexpbox_register(reg_a & reg_b, reg_c)  # c = a & b
circuit.add_classicalexpbox_register(reg_a | reg_b, reg_c)  # c = a | b

# ## Compound Logical Expressions

# Comparison operators in addition to the `==` operator are available and you can evaluate bits. Note, the `!=` operator can be useful in identifying if measurement results were trivial (for example, `meas!=0`) or not.We can operate a quantum gate on a quantum circuit when such a logical formula is satisfied as below.

# circuit.X(qreg[0], condition=reg_a[0])            # if(a[0]==1) x q[0], evaluation of a bit
circuit.X(
    qreg[0], condition=if_bit(reg_a[0])
)  # if(a[0]==1) x q[0], evaluation, same function as above
circuit.X(qreg[0], condition=if_not_bit(reg_a[0]))  # if(a[0]==0) x q[0]
circuit.X(qreg[0], condition=reg_eq(reg_a, 1))  # if(a==1) x q[0]
circuit.X(qreg[0], condition=reg_neq(reg_a, 1))  # if(a!=1) x q[0]
circuit.X(qreg[0], condition=reg_gt(reg_a, 1))  # if (reg_a > 1)
circuit.X(qreg[0], condition=reg_lt(reg_a, 1))  # if (reg_a < 1)
circuit.X(qreg[0], condition=reg_geq(reg_a, 1))  # if (reg_a >= 1)
circuit.X(qreg[0], condition=reg_leq(reg_a, 1))  # if (reg_a <= 1)

# ## Conditional Classical Assignments

# You can not only use the option `condition` to apply quantum gates but to make classical assignments as well. Note, this allows one to dynamically set flags.

circuit.add_c_setreg(1, reg_b, condition=reg_eq(reg_a, 10))  # if (a==10) b=1
circuit.add_c_setreg(1, reg_b, condition=if_not_bit(reg_a[0]))  # if (a[0]==0) b=1

# Pytket enables you to visualize where conditional operators are used within a circuit.

render_circuit_jupyter(circuit)

# <div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>
