"""
0D coronary model: two trunks off the aortic root.

    AO --[ R_LMCA ]--> LMCA territory
    AO --[ R_RCA  ]--> RCA  territory

Both trunks start at the aortic root, where pressure is prescribed from the
systemic circulation (Eq. 23), and end in myocardium, where pressure is
prescribed from the intramyocardial pressure (Eq. 24).  Both ends of both
segments are therefore boundary conditions: the system has ZERO unknowns and
each trunk is an independent Ohm's law calculation,

    Q_s = (P_AO - P_IMP_s) / R_s

The two paths meet only at the aortic root, which is pinned, so neither can
draw flow from the other.  Inter-territory steal is structurally impossible
in this configuration.

NOTE ON COMPLIANCE
------------------
C is carried in the table and assembled, but with both ends of every segment
pinned there are no free nodes for it to act on, so it has no dynamic effect
here.  The circuit is purely resistive until the tree is extended and
interior junctions appear.  The machinery is kept so that adding branches
later needs no rewrite.

TERMINAL RESISTANCE
-------------------
The paper calls each segment a four-element Windkessel: R, L, C and the
terminal impedance Z.  In the full tree only the eight leaves carry a Z --
LMCA and RCA are conduits that feed further branches, so they have none.
Truncating the tree at LMCA and RCA therefore removes everything that
provided the downstream resistance, and without a replacement the flow is
roughly an order of magnitude too high.

`terminal_resistance` supplies that replacement.  It stands in for the
microcirculation below the cut, and is given per territory because the two
beds are not equivalent: the values below are set to reproduce measured
resting flows rather than transcribed from Table 1.

Units: kPa, cm^3/s, s.  R in kPa s cm^-3, C in cm^3 kPa^-1.
"""

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


MMHG_PER_KPA = 7.50062

INLET_NODE = "AO"


# Wang et al., Table 1 -- the two trunks only.
SEGMENTS = {
    "LMCA": dict(name="Left main coronary artery",
                 R=0.2156, L=0.0227, C=0.0003,
                 node_p="AO", node_d="T_LMCA"),

    "RCA":  dict(name="Right coronary artery",
                 R=1.6671, L=0.1164, C=0.0006,
                 node_p="AO", node_d="T_RCA"),
}

SEGMENT_ORDER = ["LMCA", "RCA"]


# AHA-17 territories.  Conventional coronary distribution: the left main
# supplies the LAD and LCX territories, the right coronary the inferior wall.
# Together these are an exact partition of segments 1..17.
PERFUSION_REGIONS = {
    "LMCA": [1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17],
    "RCA":  [3, 4, 9, 10, 15],
}


# Downstream resistance standing in for the microcirculation removed by
# truncating the tree at the two trunks.
#
# Calibrated on the CYCLE-MEAN flow under pulsatile aortic pressure and a
# systolic IMP pulse -- not on the steady-state solution, because systolic
# compression lowers the mean by roughly 15% and calibrating without it
# leaves the running model short of target.
#
# Level: 250 mL/min total, the conventional resting coronary flow (about 5%
# of a 5 L/min cardiac output).  Split: 65.6% left / 34.4% right, from the
# resting per-vessel flows measured by Wieneke et al. (intracoronary Doppler
# + IVUS, n=28, angiographically smooth arteries; LAD 76.15, LCX 54.62,
# RCA 68.46, total 197.1 +/- 71.9 mL/min).  Their total is lower than 250,
# as expected for sedated supine patients, and 250 is well within their SD.
#
# These values are calibration, not measurements, and depend on mean aortic
# pressure and on the prescribed IMP.  Recalibrate when real IMP arrives.
TERMINAL_RESISTANCE = {
    "LMCA": 3.190,
    "RCA":  5.345,
}


def check_table():
    """Fail loudly on a malformed table rather than producing plausible
    numbers from a broken one."""
    errors = []

    for s in SEGMENT_ORDER:
        if s not in SEGMENTS:
            errors.append("%s is listed in SEGMENT_ORDER but not defined" % s)
    for s in SEGMENTS:
        if s not in SEGMENT_ORDER:
            errors.append("%s is defined but missing from SEGMENT_ORDER" % s)

    for s, d in SEGMENTS.items():
        for k in ("R", "C"):
            if d[k] <= 0.0:
                errors.append("%s has non-positive %s = %r" % (s, k, d[k]))
        if d["node_p"] != INLET_NODE:
            errors.append("%s does not start at the aortic root" % s)
        if d["node_p"] == d["node_d"]:
            errors.append("%s is a self-loop" % s)

    distal = [d["node_d"] for d in SEGMENTS.values()]
    if len(set(distal)) != len(distal):
        errors.append("two segments share a distal node")

    covered = sorted(sum(PERFUSION_REGIONS.values(), []))
    if covered != list(range(1, 18)):
        errors.append("perfusion territories do not partition AHA 1..17: %s"
                      % covered)
    for s in PERFUSION_REGIONS:
        if s not in SEGMENTS:
            errors.append("territory declared for unknown segment %s" % s)

    if errors:
        raise AssertionError("coronary table is malformed:\n  - "
                             + "\n  - ".join(errors))


check_table()


class CoronaryRC(object):
    """Node-based R-C coronary network.

    Pressures are the states and flows are algebraic across resistances,
    matching MyoFE's circulation model.  Nodes are either pinned (the aortic
    root and every territory outlet) or free (interior junctions, of which
    there are none in the two-trunk configuration).
    """

    def __init__(self, segment_names, dt, terminal_resistance=None):
        """terminal_resistance may be None (no downstream resistance), a
        single number applied to every territory, or a dict keyed by
        terminal segment name."""

        self.names = list(segment_names)
        for s in self.names:
            if s not in SEGMENTS:
                raise ValueError("unknown segment '%s'; available: %s"
                                 % (s, SEGMENT_ORDER))
        self.dt = dt

        nodes = set()
        for s in self.names:
            nodes.add(SEGMENTS[s]["node_p"])
            nodes.add(SEGMENTS[s]["node_d"])

        upstream = set(SEGMENTS[s]["node_p"] for s in self.names)
        self.terminal_nodes = sorted(n for n in nodes
                                     if n != INLET_NODE and n not in upstream)

        # resolve terminal resistance into a per-segment dict
        term_segs = [s for s in self.names
                     if SEGMENTS[s]["node_d"] in self.terminal_nodes]
        if terminal_resistance is None:
            self.Rt = None
        elif isinstance(terminal_resistance, dict):
            missing = [s for s in term_segs if s not in terminal_resistance]
            if missing:
                raise ValueError(
                    "terminal_resistance is missing entries for %s" % missing)
            self.Rt = dict((s, float(terminal_resistance[s]))
                           for s in term_segs)
        else:
            self.Rt = dict((s, float(terminal_resistance)) for s in term_segs)
        if self.Rt is not None:
            for s, v in self.Rt.items():
                if v <= 0.0:
                    raise ValueError(
                        "terminal_resistance for %s must be positive, got %r"
                        % (s, v))

        if self.Rt is None:
            self.free_nodes = sorted(n for n in nodes
                                     if n != INLET_NODE
                                     and n not in self.terminal_nodes)
        else:
            self.free_nodes = sorted(n for n in nodes if n != INLET_NODE)

        self.idx = dict((n, i) for i, n in enumerate(self.free_nodes))
        self.n = len(self.free_nodes)

        self.terminal_segment = dict(
            (SEGMENTS[s]["node_d"], s) for s in self.names
            if SEGMENTS[s]["node_d"] in self.terminal_nodes)
        self.terminals = [self.terminal_segment[n] for n in self.terminal_nodes]

        self.C = dict((n, 0.0) for n in nodes)
        for s in self.names:
            self.C[SEGMENTS[s]["node_d"]] += SEGMENTS[s]["C"]

        self._assemble()

    def _pinned(self, node, P_AO, P_IMP):
        if node == INLET_NODE:
            return P_AO
        return P_IMP[self.terminal_segment[node]]

    def _assemble(self):
        A = np.zeros((self.n, self.n))
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
                s = self.terminal_segment[nd]
                A[self.idx[nd], self.idx[nd]] += 1.0 / self.Rt[s]
        self.A = A
        if _HAVE_SCIPY and self.n > 0:
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
                s = self.terminal_segment[nd]
                b[self.idx[nd]] += P_IMP[s] / self.Rt[s]
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
        """Flow through each segment, positive from aorta into myocardium."""
        q = {}
        for s in self.names:
            pa = self.pressure(P, SEGMENTS[s]["node_p"], P_AO, P_IMP)
            pd = self.pressure(P, SEGMENTS[s]["node_d"], P_AO, P_IMP)
            q[s] = (pa - pd) / SEGMENTS[s]["R"]
        return q

    def perfusion(self, P, P_AO, P_IMP):
        """Flow delivered into each territory."""
        if self.Rt is None:
            q = self.flows(P, P_AO, P_IMP)
            return dict((s, q[s]) for s in self.terminals)
        out = {}
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            out[s] = (P[self.idx[nd]] - P_IMP[s]) / self.Rt[s]
        return out

    def describe(self):
        print("  segments  : %d  %s" % (len(self.names), ", ".join(self.names)))
        print("  free nodes: %d  %s"
              % (self.n, ", ".join(self.free_nodes) if self.n else "(none)"))
        print("  pinned    : %-8s = P_AO" % INLET_NODE)
        for nd in self.terminal_nodes:
            print("              %-8s = P_IMP[%s]"
                  % (nd, self.terminal_segment[nd]))
        if self.Rt is not None:
            print("  terminal R: %s"
                  % ", ".join("%s=%.4g" % (s, self.Rt[s])
                              for s in sorted(self.Rt)))


SUBTREES = {"lmca_rca": SEGMENT_ORDER}


if __name__ == "__main__":

    print("Coronary segments")
    print("=" * 62)
    print("%-6s %9s %10s   %-6s -> %-8s  AHA territory"
          % ("seg", "R", "C", "prox", "dist"))
    for s in SEGMENT_ORDER:
        d = SEGMENTS[s]
        print("%-6s %9.4f %10.5f   %-6s -> %-8s  %s"
              % (s, d["R"], d["C"], d["node_p"], d["node_d"],
                 PERFUSION_REGIONS[s]))

    print()
    m = CoronaryRC(SEGMENT_ORDER, 1.0e-3)
    m.describe()

    print()
    print("Steady state, P_AO = 90 mmHg, IMP = 0")
    print("=" * 62)
    P_AO = 90.0 / MMHG_PER_KPA
    P_IMP = dict((s, 0.0) for s in m.terminals)
    P = m.steady_state(P_AO, P_IMP)
    q = m.flows(P, P_AO, P_IMP)
    total = 0.0
    for s in m.terminals:
        R = SEGMENTS[s]["R"]
        print("  %-5s Q = (%.4f - 0)/%.4f = %8.3f cm3/s = %6.0f mL/min"
              % (s, P_AO, R, q[s], q[s] * 60.0))
        total += q[s]
    print("  total%29.3f cm3/s = %6.0f mL/min" % (total, total * 60.0))
    print("  physiological whole-heart coronary flow is roughly 250 mL/min")
