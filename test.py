from pytket import Circuit, OpType
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.passes import *

b = QuantinuumBackend("H1-1SC")
c = Circuit(2)
c.H(0)
c.CX(0, 1)
c.Rz(0.2, 1)
c.CX(0, 1)


# c.add_gate(OpType.TK1, [0, 0.5, 0], [0])
# c.add_gate(OpType.TK1, [1, 0.5, 0], [2])
# c.add_gate(OpType.TK2, [0.5, 0, 0], [0, 2])
# c.add_gate(OpType.TK1, [0, 0.5, 0], [0])
# c.add_gate(OpType.TK1, [1, 0.5, 1.2], [2])
# c.add_gate(OpType.TK2, [0.5, 0, 0], [0, 2])
# c.add_gate(OpType.TK1, [0.5, 0.5, 0.5], [0])
# c.add_gate(OpType.TK1, [0, 0, 1], [2])

SynthesiseTK().apply(c)
print(c.get_commands())
print(10*"=", " (SynthesiseTK)")
KAKDecomposition(target_2qb_gate=OpType.TK2).apply(c)
print(c.get_commands())
print(10*"=", " (KAK)")
# CliffordSimp(False).apply(c)
# print(c.get_commands())
# print(10*"=", " (Clifford Simp)")
# KAKDecomposition(target_2qb_gate=OpType.TK2).apply(c)
# print(c.get_commands())
# print(10*"=", " (KAK)")
# SynthesiseTK().apply(c)
# print(c.get_commands())
# print(10*"=", " (SynthesiseTK)")
# ThreeQubitSquash().apply(c)
# print(c.get_commands())
# print(10*"=", " (ThreeQbSquash)")
# CliffordSimp(False).apply(c)
# print(c.get_commands())
# print(10*"=", " (Clifford Simp)")
# KAKDecomposition(target_2qb_gate=OpType.TK2).apply(c)
# print(c.get_commands())
# print(10*"=", " (KAK)")
# SynthesiseTK().apply(c)
# print(c.get_commands())
# print(10*"=", " (SynthesiseTK)")




# =======================================

# c.add_gate(OpType.TK1, [2, 1.5, 0], [0])
# c.add_gate(OpType.TK1, [0.5, 0.5, 0], [2])
# c.add_gate(OpType.TK2, [0.2, 0, 0], [0, 2])
# c.add_gate(OpType.TK1, [0, 3.5, 3.5], [0])
# c.add_gate(OpType.TK1, [0, 3.5, 3.5], [2])
# c.measure_all()


# fidelities = {"ZZMax_fidelity": 1.0}
# _xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])

# FullPeepholeOptimise(target_2qb_gate=OpType.TK2).apply(c)
# print(c.get_commands())
# print(10*"=", " (FullPeephole)")
# NormaliseTK2().apply(c)
# DecomposeTK2(**fidelities).apply(c)
# print(c.get_commands())
# print(10*"=", " (NormaliseTK2 + DecomposeTK2)")
# auto_rebase_pass({OpType.ZZMax, OpType.Rz, OpType.PhasedX}).apply(c)
# print(c.get_commands())
# print(10*"=", " (Rebase)")
# RemoveRedundancies().apply(c)
# auto_squash_pass({OpType.PhasedX, OpType.Rz}).apply(c)
# print(c.get_commands())
# print(10*"=", " (Squash)")
# SimplifyInitial(allow_classical=False, create_all_qubits=True, xcirc=_xcirc).apply(c)
# print(c.get_commands())
# print(10*"=", " (SimplifyInitial)")

