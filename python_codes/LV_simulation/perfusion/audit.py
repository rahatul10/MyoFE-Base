import numpy as np
from coronary_segments import SEGMENTS
from coronary_rc import CoronaryRC, SUBTREES

np.set_printoptions(precision=6, suppress=True)
dt = 1e-3
m = CoronaryRC(SUBTREES["branch"], dt)

print("SUBTREE: LMCA -> LAD -> {LAD1, LAD2}")
print("free nodes (matrix row/col order):", m.free_nodes)
print()
print("R values:")
for s in m.names:
    print("   R_%-5s = %.4f   1/R = %.6f" % (s, SEGMENTS[s]["R"], 1.0/SEGMENTS[s]["R"]))
print()
print("C at each free node (distal placement):")
for n in m.free_nodes:
    print("   C[%-7s] = %.6f   C/dt = %.6f" % (n, m.C[n], m.C[n]/dt))
print()
print("MATRIX A (backward Euler):")
print(m.A)
print()
print("RHS boundary load b_bc for P_AO=12, P_IMP=2 everywhere:")
print(m._bc_load(12.0, {"LAD1":2.0,"LAD2":2.0}))
print()
print("STEADY-STATE matrix (A with C/dt removed):")
As = m.A.copy()
for n in m.free_nodes:
    As[m.idx[n], m.idx[n]] -= m.C[n]/dt
print(As)
print()
P = np.linalg.solve(As, m._bc_load(12.0, {"LAD1":2.0,"LAD2":2.0}))
for n in m.free_nodes:
    print("   P[%-7s] = %.6f kPa" % (n, P[m.idx[n]]))
