"""
0D epicardial coronary network, R-C form, no terminal impedance.

Structure mirrors MyoFE's circulation model (circulation.py): pressures are
the states, flows are algebraic across resistances.  Inductance is dropped.
The terminal impedance Z of Table 1 is NOT used -- see note below.

States live at NODES:

    AO                pinned to P_AO          (Eq. 23)
    terminal nodes    pinned to P_IMP^s       (Eq. 24)
    interior nodes    free, one state each

    C_n dP_n/dt = sum(inflows) - sum(outflows)
    Q_s = (P_{node_p} - P_{node_d}) / R_s

Backward Euler gives A P^{n+1} = b with A constant -> factorize once.
Unknowns = number of interior nodes (7 for the full tree).

ON Z
----
Eq. (24) sets the outlet pressure equal to the averaged intramyocardial
pressure directly; Z does not appear in it.  Section 2.1 states only that Z
is determined primarily by the downstream microcirculation, represented with
the structured tree model of Olufsen, adopted from Cai et al.  A structured
tree terminal impedance is a frequency-domain quantity Z(omega), not a
series resistor, and the manuscript never states how the Table 1 values
enter the discrete system.  Treating them as resistors is an assumption the
paper does not license, so Z is skipped here.

Consequence: with no downstream resistance, terminal flows are set by the
epicardial resistances alone and come out well above physiological.  See the
flow comparison printed by the test script.  Revisit once the authors
clarify, or replace with a lumped terminal resistance calibrated to a target
flow.

COMPLIANCE PLACEMENT
--------------------
Fig. 3(b) draws the capacitance at the distal end but notes the topology
depends on the choice of alpha and beta; the paper uses alpha = beta = 1/2,
which puts the segment average mid-segment.  'distal' assigns C_s to node_d,
'split' assigns half to each end.  With 'distal', the compliance of a
terminal segment sits on a pinned node and has no dynamic effect.

Units: kPa, cm^3/s, s.
"""

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from coronary_segments import SEGMENTS, INLET_NODE


MMHG_PER_KPA = 7.50062


SUBTREES = {
    "single": ["LMCA"],
    "series": ["LMCA", "LAD"],
    "branch": ["LMCA", "LAD", "LAD1", "LAD2"],
    "lad3":   ["LMCA", "LAD", "LAD1", "LAD2", "LAD3", "LAD4"],
    "full":   list(SEGMENTS.keys()),
}


# AHA-17 territories for the LAD terminals (Wang et al., p. 13).  Duplicated
# here only so this file runs standalone; production code must import
# PERFUSION_REGIONS from dependencies/aha_segmentation.py.
PERFUSION_REGIONS_LAD = {
    "LAD1": [1, 2, 8],
    "LAD3": [13, 14, 17],
    "LAD4": [7],
}


class CoronaryRC(object):

    def __init__(self, segment_names, dt, compliance_placement="distal",
                 terminal_resistance=None):
        if compliance_placement not in ("distal", "split"):
            raise ValueError("compliance_placement must be 'distal' or 'split'")
        self.names = list(segment_names)
        self.dt = dt
        self.placement = compliance_placement
        self.Rt = terminal_resistance

        node_owner = {}
        for s in self.names:
            node_owner[SEGMENTS[s]["node_d"]] = s
        for s in self.names:
            pnode = SEGMENTS[s]["node_p"]
            if pnode != INLET_NODE and pnode not in node_owner:
                raise ValueError(
                    "segment %s needs its parent at node %s, which is not in "
                    "the selected subtree" % (s, pnode))

        upstream = set(SEGMENTS[s]["node_p"] for s in self.names)
        nodes = set()
        for s in self.names:
            nodes.add(SEGMENTS[s]["node_p"])
            nodes.add(SEGMENTS[s]["node_d"])

        self.terminal_nodes = sorted(n for n in nodes
                                     if n != INLET_NODE and n not in upstream)
        if self.Rt is None:
            self.free_nodes = sorted(n for n in nodes
                                     if n != INLET_NODE
                                     and n not in self.terminal_nodes)
        else:
            self.free_nodes = sorted(n for n in nodes if n != INLET_NODE)
        self.idx = dict((n, i) for i, n in enumerate(self.free_nodes))
        self.n = len(self.free_nodes)

        # segment feeding each terminal node: its flow is the perfusion
        # quantity Q that will feed Eq. 19
        self.terminal_segment = dict(
            (SEGMENTS[s]["node_d"], s) for s in self.names
            if SEGMENTS[s]["node_d"] in self.terminal_nodes)
        self.terminals = [self.terminal_segment[n] for n in self.terminal_nodes]

        self.C = dict((n, 0.0) for n in nodes)
        for s in self.names:
            c = SEGMENTS[s]["C"]
            if self.placement == "distal":
                self.C[SEGMENTS[s]["node_d"]] += c
            else:
                self.C[SEGMENTS[s]["node_d"]] += 0.5 * c
                self.C[SEGMENTS[s]["node_p"]] += 0.5 * c

        self._assemble()

    def _pinned(self, node, P_AO, P_IMP):
        if node == INLET_NODE:
            return P_AO
        return P_IMP[self.terminal_segment[node]]

    def _assemble(self):
        n = self.n
        A = np.zeros((n, n))
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] += self.C[node] / self.dt
        for s in self.names:
            a, b = SEGMENTS[s]["node_p"], SEGMENTS[s]["node_d"]
            g = 1.0 / SEGMENTS[s]["R"]
            for u, v in ((a, b), (b, a)):
                if u in self.idx:
                    A[self.idx[u], self.idx[u]] += g
                    if v in self.idx:
                        A[self.idx[u], self.idx[v]] -= g
        if self.Rt is not None:
            for nd in self.terminal_nodes:
                A[self.idx[nd], self.idx[nd]] += 1.0 / self.Rt
        self.A = A
        if _HAVE_SCIPY and n > 0:
            self._lu = lu_factor(A)

    def _bc_load(self, P_AO, P_IMP):
        b = np.zeros(self.n)
        for s in self.names:
            a, d = SEGMENTS[s]["node_p"], SEGMENTS[s]["node_d"]
            g = 1.0 / SEGMENTS[s]["R"]
            for u, v in ((a, d), (d, a)):
                if u in self.idx and v not in self.idx:
                    b[self.idx[u]] += g * self._pinned(v, P_AO, P_IMP)
        if self.Rt is not None:
            for nd in self.terminal_nodes:
                b[self.idx[nd]] += (P_IMP[self.terminal_segment[nd]]
                                    / self.Rt)
        return b

    def step(self, P_prev, P_AO, P_IMP):
        if self.n == 0:
            return np.zeros(0)
        b = self._bc_load(P_AO, P_IMP)
        for node in self.free_nodes:
            i = self.idx[node]
            b[i] += self.C[node] / self.dt * P_prev[i]
        if _HAVE_SCIPY:
            return lu_solve(self._lu, b)
        return np.linalg.solve(self.A, b)

    def steady_state(self, P_AO, P_IMP):
        if self.n == 0:
            return np.zeros(0)
        A = self.A.copy()
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] -= self.C[node] / self.dt
        return np.linalg.solve(A, self._bc_load(P_AO, P_IMP))

    def pressure(self, P, node, P_AO, P_IMP):
        if node in self.idx:
            return P[self.idx[node]]
        return self._pinned(node, P_AO, P_IMP)

    def flows(self, P, P_AO, P_IMP):
        """Flow through every segment, positive proximal to distal."""
        q = {}
        for s in self.names:
            pa = self.pressure(P, SEGMENTS[s]["node_p"], P_AO, P_IMP)
            pd = self.pressure(P, SEGMENTS[s]["node_d"], P_AO, P_IMP)
            q[s] = (pa - pd) / SEGMENTS[s]["R"]
        return q

    def perfusion(self, P, P_AO, P_IMP):
        """Flow actually delivered into each territory.  With no terminal
        resistance this equals the terminal segment flow; with one it is the
        flow across that resistance."""
        if self.Rt is None:
            q = self.flows(P, P_AO, P_IMP)
            return dict((s, q[s]) for s in self.terminals)
        out = {}
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            out[s] = (P[self.idx[nd]] - P_IMP[s]) / self.Rt
        return out

    def describe(self):
        print("  segments  : %d  %s" % (len(self.names), ", ".join(self.names)))
        print("  free nodes: %d  %s"
              % (self.n, ", ".join(self.free_nodes) if self.n else "(none)"))
        print("  pinned    : %-9s = P_AO" % INLET_NODE)
        for nd in self.terminal_nodes:
            print("              %-9s = P_IMP[%s]"
                  % (nd, self.terminal_segment[nd]))
        print("  compliance: %s placement" % self.placement)
        if self.Rt is not None:
            print("  terminal R: %.4g (uniform, diagnostic only)"
                  % self.Rt)
