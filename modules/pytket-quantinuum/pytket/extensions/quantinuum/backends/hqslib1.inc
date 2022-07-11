// file: hqslib1.inc

// --- Primitives ---

// 1-qubit Z-axis rotation
// NOTE: For HQS compiler / simulation, only one of these can be active at a time
opaque Rz(lambda) q;
//gate Rz(lambda) q
//{
//   U(0,0,lambda) q;
//}

// 1-qubit rotation gate
// NOTE: For HQS compiler / simulation, only one of these can be active at a time
opaque U1q(theta, phi) q;
//gate U1q(theta, phi) q
//{
//   U(theta, phi, 0) q;
//}

// Unitary 2-qubit gate
// NOTE: For HQS compiler / simulation, only one of these can be active at a time
opaque ZZ() q1,q2;
//gate ZZ() q1,q2
//{
//	U1q(pi/2, pi/2) q2;
//	CX q1, q2;
//	Rz(pi/2) q1;
//	U1q(pi/2, 0) q2;
//	U1q(pi/2, -pi/2) q2;
//}

// --- Standard Gates ---

// Pauli gate: bit-flip
gate x() a
{
   U1q(pi, 0) a;
}
// Pauli gate: bit and phase flip
gate y() a
{
   U1q(pi, pi/2) a;
}
// Pauli gate: phase flip
gate z() a
{
   Rz(pi) a;
}
// Clifford gate: CNOT
gate CX() c,t
{
   U1q(-pi/2, pi/2) t;
   ZZ() c, t;
   Rz(-pi/2) c;
   U1q(pi/2, pi) t;
   Rz(-pi/2) t;
}

gate cx() c,t
{
   CX c,t;
}
// Clifford gate: Hadamard
gate h() a
{
   U1q(pi/2, -pi/2) a;
   Rz(pi) a;
}
// Clifford gate: sqrt(Z) phase gate
gate s() a
{
   Rz(pi/2) a;
}
// Clifford gate: conjugate of sqrt(Z)
gate sdg() a
{
   Rz(-pi/2) a;
}
// C3 gate: sqrt(S) phase gate
gate t() a
{
   Rz(pi/4) a;
}
// C3 gate: conjugate of sqrt(S)
gate tdg() a
{
   Rz(-pi/4) a;
}

// --- Standard rotations ---

// Rotation around X-axis
gate rx(theta) a
{
   U1q(theta, 0) a;
}
// Rotation around Y-axis
gate ry(theta) a
{
   U1q(theta, pi/2) a;
}
// Rotation around Z-axis
gate rz(phi) a
{
   Rz(phi) a;
}

// --- QE Standard User-Defined Gates  ---

// controlled-Phase
gate cz() a,b
{
   h b;
   cx a,b;
   h b;
}
// controlled-Y
gate cy() a,b
{
   sdg b;
   cx a,b;
   s b;
}
// controlled-H
gate ch() a,b
{
   h b; sdg b;
   cx a,b;
   h b; t b;
   cx a,b;
   t b; h b; s b; x b; s a;
}
// C3 gate: Toffoli
gate ccx() a,b,c
{
   h c;
   cx b,c; tdg c;
   cx a,c; t c;
   cx b,c; tdg c;
   cx a,c; t b; t c; h c;
   cx a,b; t a; tdg b;
   cx a,b;
}
// controlled rz rotation
gate crz(lambda) a,b
{
   Rz(lambda/2) b;
   cx a, b;
   Rz(-lambda/2) b;
   cx a, b;
}
// controlled phase rotation
gate cu1(lambda) a,b
{
   Rz(lambda/2) a;
   cx a, b;
   Rz(-lambda/2) b;
   cx a, b;
   Rz(lambda/2) b;
}
// controlled-U
gate cu3(theta, phi, lambda) c, t
{
   Rz((lambda-phi)/2) t;
   cx c, t;
   Rz(-(phi+lambda)/2) t;
   U1q(-theta/2, pi/2) t;
   cx c, t;
   U1q(theta/2, pi/2) t;
   Rz(phi) t;
}
