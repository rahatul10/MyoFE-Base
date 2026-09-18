"""
0D epicardial coronary network, R-C form.

Currently three perfused trunks:

    AO --[ R_LMCA ]-- N_LMCA --[ R_LAD ]-- T_LAD --[ Rt_LAD ]-- P_IMP_LAD
                             --[ R_LCX ]-- T_LCX --[ Rt_LCX ]-- P_IMP_LCX
    AO --[ R_RCA  ]--------------------- T_RCA --[ Rt_RCA ]-- P_IMP_RCA

The aortic root is pinned to P_AO from the systemic circulation (Eq. 23) and
every territory outlet is pinned to the averaged intramyocardial pressure of
its territory (Eq. 24).  Everything between them is solved:

    C_n dP_n/dt = sum(inflows) - sum(outflows)
    Q_s = (P_node_p - P_node_d) / R_s

Backward Euler gives A P^{n+1} = b with A constant in time, so the matrix is
factorized once.  Unknowns = interior junctions + terminal nodes.

STRUCTURE AND STEAL
-------------------
LAD and LCX meet at N_LMCA, so they are hydraulically coupled and flow could
in principle move sideways between them if their territory pressures diverge
far enough.  It does not here: the terminal resistances are an order of
magnitude larger than the epicardial ones, so the pressure drop across the
microcirculation dominates and both territories stay forward-flowing
throughout the cycle.  This is checked in the __main__ block; re-check it
whenever a branch is added, because it is exactly the failure that appeared
in the earlier six-segment LAD subtree.

TERMINAL RESISTANCE
-------------------
The paper calls each segment a four-element Windkessel: R, L, C and the
terminal impedance Z.  In the full tree only the eight leaves carry a Z --
LMCA, LAD, LCX and RCA are conduits feeding further branches, so they have
none.  Truncating the tree therefore removes everything that provided the
downstream resistance, and without a replacement the flow is roughly an
order of magnitude too high.

TERMINAL_RESISTANCE supplies that replacement.  It stands in for the
microcirculation below the cut and is given per territory because the beds
are not equivalent.  The values are calibration, not measurement: they are
set to reproduce measured resting flows and are NOT Table 1's Z.

INDUCTANCE
----------
Set inertance=True (JSON: "inertance") to solve the full R-L-C system of
Wang et al. Eq. 3 rather than the R-C reduction.

With L the segment flows become states alongside the node pressures:

    L_s dQ_s/dt + R_s Q_s = P_p - P_d          one per segment
    C_n dP_n/dt = sum(Q_in) - sum(Q_out)       one per free node

Backward Euler on both gives a single constant linear system in [P; Q],
factorized once.  Setting every L to zero reproduces the R-C solver exactly.

Expect little difference.  Over a periodic cycle the integral of L dQ/dt is
exactly zero, so the cycle mean cannot change; and the waveform barely moves
either, because the terminal resistances (72-170) dwarf the epicardial ones
(0.2-3.6).  At typical flow the whole epicardial path drops under 1 mmHg
against about 29 mmHg across the terminal resistance, so inertance acts on a
part of the circuit that carries almost none of the pressure.

UNITS
-----
mmHg, mL/s, s.  R and Rt in mmHg s mL^-1, C in mL mmHg^-1, L in mmHg s^2
mL^-1.  This matches MyoFE's circulation model, so no conversion happens
anywhere in this module.

Wang et al.'s Table 1 labels these values kPa s cm^-3, but that label is
almost certainly wrong.  The table derives from Duanmu et al., Sci Rep
8:874 (2018), whose equivalent table is explicitly in mmHg s mL^-1 with
values of the same magnitude (their RCA 1.64 against 1.6671 here; PDA 2.31
against 2.1974; PLA 1.31 against 1.2531) -- and RCA anatomy does not vary
by a factor of 7.5 between patients.  Three checks agree:

  1. Read as kPa, the full 16-segment tree with Table 1's Z gives 52.6
     mL/min, about a fifth of physiological.  Read as mmHg it gives 395.
  2. Wang's own reported peak flows (LAD1 0.669, LAD4 1.205 mL/s) are
     reproduced to 11% by the kPa reading and to 80-86% by the mmHg
     reading, the remainder being peak-versus-steady-state.
  3. Vessel lengths implied by R and L under Poiseuille are absurd in kPa
     (LMCA 187 mm, RCA 636 mm) and anatomical in mmHg (25 mm, 85 mm).

Python 2.7 compatible.
"""

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


INLET_NODE = "AO"


# ---------------------------------------------------------------------------
# Segment table -- Wang et al., Table 1, verified against the rendered page.
# ---------------------------------------------------------------------------

SEGMENTS = {
    "LMCA": dict(name="Left main coronary artery",
                 R=0.2156, L=0.0227, C=0.0003,
                 node_p="AO",     node_d="N_LMCA"),

    "LAD":  dict(name="Left anterior descending artery",
                 R=0.4299, L=0.0294, C=0.0002,
                 node_p="N_LMCA", node_d="N_LAD"),

    "LAD1": dict(name="First LAD branch",
                 R=0.4944, L=0.0327, C=0.00020, Z=145.60,
                 node_p="N_LAD",  node_d="T_LAD1"),

    "LAD2": dict(name="LAD continuation",
                 R=2.3796, L=0.1107, C=0.00030,
                 node_p="N_LAD",  node_d="N_LAD2"),

    "LAD3": dict(name="Distal LAD / apical branch",
                 R=3.1904, L=0.1058, C=0.00010, Z=155.00,
                 node_p="N_LAD2", node_d="T_LAD3"),

    "LAD4": dict(name="Second diagonal branch",
                 R=3.6283, L=0.0772, C=0.00001, Z=80.20,
                 node_p="N_LAD2", node_d="T_LAD4"),

    "LCX":  dict(name="Left circumflex artery",
                 R=0.3390, L=0.0230, C=0.0001,
                 node_p="N_LMCA", node_d="N_LCX"),

    "MARG1": dict(name="First obtuse marginal branch",
                  R=1.6994, L=0.0666, C=0.0001, Z=155.00,
                  node_p="N_LCX",  node_d="T_MARG1"),

    "LCX1": dict(name="Circumflex continuation, first segment",
                 R=0.4215, L=0.0224, C=0.0001,
                 node_p="N_LCX",  node_d="N_LCX1"),

    "MARG2": dict(name="Second obtuse marginal branch",
                  R=2.5890, L=0.0761, C=0.0001, Z=168.40,
                  node_p="N_LCX1", node_d="T_MARG2"),

    "LCX2": dict(name="Circumflex continuation, second segment",
                 R=0.8890, L=0.0364, C=0.0001,
                 node_p="N_LCX1", node_d="N_LCX2"),

    "MARG3": dict(name="Third obtuse marginal branch",
                  R=2.7165, L=0.0866, C=0.0001, Z=72.00,
                  node_p="N_LCX2", node_d="T_MARG3"),

    "LCX3": dict(name="Distal circumflex",
                 R=2.9208, L=0.0922, C=0.0001, Z=170.10,
                 node_p="N_LCX2", node_d="T_LCX3"),

    "RCA":  dict(name="Right coronary artery",
                 R=1.6671, L=0.1164, C=0.0006,
                 node_p="AO",     node_d="N_RCA"),

    "PLA":  dict(name="Posterolateral artery (right ventricle)",
                 R=1.2531, L=0.0627, C=0.0002, Z=59.81,
                 node_p="N_RCA",  node_d="T_PLA"),

    "PDA":  dict(name="Posterior descending artery",
                 R=2.1974, L=0.0779, C=0.0001, Z=139.72,
                 node_p="N_RCA",  node_d="T_PDA"),
}

SEGMENT_ORDER = ["LMCA", "LAD", "LAD1", "LAD2", "LAD3", "LAD4",
                 "LCX", "MARG1", "LCX1", "MARG2", "LCX2", "MARG3", "LCX3",
                 "RCA", "PLA", "PDA"]


# ---------------------------------------------------------------------------
# Configurations the solver can be built on.
# ---------------------------------------------------------------------------
# 'lad_lcx_rca' is the production configuration.  'lmca_rca' is kept as the
# simpler fallback: LMCA terminates directly, so LAD and LCX are dropped.

SUBTREES = {
    "lmca_rca":    ["LMCA", "RCA"],
    "lad_lcx_rca": ["LMCA", "LAD", "LCX", "RCA"],
    "lad12_lcx_rca": ["LMCA", "LAD", "LAD1", "LAD2", "LCX", "RCA"],
    "lad_full_lcx_rca": ["LMCA", "LAD", "LAD1", "LAD2", "LAD3", "LAD4",
                         "LCX", "RCA"],
    "full": SEGMENT_ORDER,
}

# 'lmca_rca' needs LMCA to be a leaf, which contradicts the table above where
# it feeds N_LMCA.  Rather than carry two tables, that configuration rewires
# LMCA's distal node when it is selected; see _resolve_segments.
_LEAF_OVERRIDE = {
    "lmca_rca":       {"LMCA": "T_LMCA", "LCX": "T_LCX", "RCA": "T_RCA"},
    "lad_lcx_rca":    {"LAD": "T_LAD", "LCX": "T_LCX", "RCA": "T_RCA"},
    "lad12_lcx_rca":  {"LAD2": "T_LAD2", "LCX": "T_LCX", "RCA": "T_RCA"},
    "lad_full_lcx_rca": {"LCX": "T_LCX", "RCA": "T_RCA"},
    # "full" needs no override -- every segment sits where Table 1 puts it.
}


# ---------------------------------------------------------------------------
# AHA-17 perfusion territories, per configuration.
# ---------------------------------------------------------------------------
# These follow Wang et al. Fig. 5, which gives the branch-level correspondence
# for THIS patient's reconstructed anatomy:
#
#   LAD1  1, 2, 8      MARG1  6, 12      LCX3  4, 10
#   LAD4  7            MARG2  5, 11      PDA   3, 9
#   LAD3  13, 14, 17   MARG3  15, 16
#
# The maps below are that table rolled up to whichever branches a given
# configuration actually terminates in.  Note this is NOT the textbook
# right-dominant distribution: in this patient the circumflex supplies the
# inferior wall (LCX3 takes 4 and 10, MARG3 takes 15), leaving the PDA with
# only the two inferoseptal segments.  Do not "correct" it to the standard
# Cerqueira assignment -- it is patient-specific anatomy from their CT, and
# it must match dependencies/aha_segmentation.py, which is authoritative.

PERFUSION_REGIONS = {
    "lmca_rca": {
        "LMCA": [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17],
        "RCA":  [3, 9],
    },
    "lad_lcx_rca": {
        "LAD":  [1, 2, 7, 8, 13, 14, 17],
        "LCX":  [4, 5, 6, 10, 11, 12, 15, 16],
        "RCA":  [3, 9],
    },
    "lad12_lcx_rca": {
        "LAD1": [1, 2, 8],
        "LAD2": [7, 13, 14, 17],
        "LCX":  [4, 5, 6, 10, 11, 12, 15, 16],
        "RCA":  [3, 9],
    },
    "lad_full_lcx_rca": {
        "LAD1": [1, 2, 8],
        "LAD3": [13, 14, 17],
        "LAD4": [7],
        "LCX":  [4, 5, 6, 10, 11, 12, 15, 16],
        "RCA":  [3, 9],
    },
    # The full tree is Wang Fig. 5 verbatim.  PLA has NO AHA segments: it
    # supplies right ventricle, which this LV-only mesh does not contain.
    # See EXTRA_LV_TERMINALS below for how its IMP is handled.
    "full": {
        "LAD1":  [1, 2, 8],
        "LAD4":  [7],
        "LAD3":  [13, 14, 17],
        "MARG1": [6, 12],
        "MARG2": [5, 11],
        "MARG3": [15, 16],
        "LCX3":  [4, 10],
        "PDA":   [3, 9],
        "PLA":   [],
    },
}


# ---------------------------------------------------------------------------
# Terminals that perfuse tissue outside the LV mesh
# ---------------------------------------------------------------------------
# PLA supplies the right ventricle.  There is no RV in this mesh, so its IMP
# cannot be averaged from any territory.  Wang et al. take IMP in the PLA
# region as roughly one third of the LV wall value, as a surrogate for RV
# intramyocardial pressure; that fraction is applied to the volume-weighted
# mean IMP over all LV territories.
#
# These terminals still carry flow and still load the tree -- PLA takes about
# 80 mL/min, which is a real part of the coronary circulation.  They are just
# excluded from the AHA 1..17 partition check, since they own no LV segments.

EXTRA_LV_TERMINALS = {
    "full": {"PLA": 1.0 / 3.0},
}


# ---------------------------------------------------------------------------
# Terminal resistances, per configuration.
# ---------------------------------------------------------------------------
# Calibrated on the CYCLE-MEAN flow driven by the aortic pressure recorded
# in a completed run (circ.data['pressure_aorta'], mean 86.6 mmHg over the
# converged window, baroreflex inactive) with the prescribed systolic IMP
# active -- not on the steady state, because systolic compression lowers the
# mean by roughly 15%.
#
# Sanity check on the magnitudes: Duanmu et al. report per-terminal values
# of 47-290 mmHg s mL^-1 for nine terminals, validated against measured
# perfusion resistance.  Three of those in parallel is around 50, which is
# where these land.  They did not, under the earlier kPa reading.
#
# NOTE the inlet is the aorta compartment, not arteries.  The coronary ostia
# sit in the aortic root, upstream of the systemic arterial resistance; the
# two differ by about 1.5 mmHg and these values assume the former.
#
# Level: 250 mL/min total, the conventional resting coronary flow (about 5%
# of a 5 L/min cardiac output).
#
# Split: the resting per-vessel flows measured by Wieneke et al.
# (intracoronary Doppler + IVUS, n=28, angiographically smooth arteries):
# LAD 76.15 +/- 33.41, LCX 54.62 +/- 24.59, RCA 68.46 +/- 31.87 mL/min,
# total 197.1 +/- 71.9.  Their total is lower than 250, as expected for
# sedated supine patients, and 250 sits well within one SD.  The proportions
# 38.2 / 27.4 / 34.4 % are taken from their per-vessel means and scaled.
#
# These depend on mean aortic pressure and on the prescribed IMP.
# Recalibrate when real IMP arrives (Stage 3).

TERMINAL_RESISTANCE = {
    "lmca_rca": {
        "LMCA": 23.832,
        "RCA":  48.410,
    },
    "lad_lcx_rca": {
        "LAD":  40.496,
        "LCX":  56.720,
        "RCA":  48.410,
    },
    "lad12_lcx_rca": {
        "LAD1": 145.60,
        "LAD2":  53.623,
        "LCX":   56.720,
        "RCA":   48.410,
    },
    # The whole LAD subtree now runs on Wang's published Z: LAD1, LAD3 and
    # LAD4 are all leaves in Table 1 and all carry one.  Nothing in the LAD
    # territory is fitted.
    "lad_full_lcx_rca": {
        "LAD1": 145.60,
        "LAD3": 155.00,
        "LAD4":  80.20,
        "LCX":   56.720,
        "RCA":   48.410,
    },
    # The full tree needs NO calibration at all: every terminal is a leaf in
    # Table 1 and carries its own published Z.  Nothing here is fitted.
    "full": {
        "LAD1":  145.60,
        "LAD3":  155.00,
        "LAD4":   80.20,
        "MARG1": 155.00,
        "MARG2": 168.40,
        "MARG3":  72.00,
        "LCX3":  170.10,
        "PLA":    59.81,
        "PDA":   139.72,
    },
}


# ---------------------------------------------------------------------------
# Configuration resolution and validation
# ---------------------------------------------------------------------------

def _resolve_segments(subtree):
    """Return {name: segment dict} for a configuration, applying any leaf
    override.  The returned dicts are copies, so SEGMENTS is never mutated."""
    if subtree not in SUBTREES:
        raise ValueError("unknown coronary subtree '%s'; options are %s"
                         % (subtree, sorted(SUBTREES.keys())))
    override = _LEAF_OVERRIDE.get(subtree, {})
    out = {}
    for s in SUBTREES[subtree]:
        d = dict(SEGMENTS[s])
        if s in override:
            d["node_d"] = override[s]
        out[s] = d
    return out


def terminals_of(subtree):
    """Segment names that end in myocardium for this configuration."""
    segs = _resolve_segments(subtree)
    upstream = set(d["node_p"] for d in segs.values())
    return [s for s in SUBTREES[subtree]
            if segs[s]["node_d"] not in upstream]


def check_tables(verbose=False):
    """Prove every declared configuration is a well-formed tree with a
    complete, consistent territory map and terminal resistances.  A malformed
    table should fail on import, not produce plausible-looking waveforms."""
    errors = []

    for s in SEGMENT_ORDER:
        if s not in SEGMENTS:
            errors.append("%s is in SEGMENT_ORDER but not defined" % s)
    for s in SEGMENTS:
        if s not in SEGMENT_ORDER:
            errors.append("%s is defined but missing from SEGMENT_ORDER" % s)
        for k in ("R", "C"):
            if SEGMENTS[s][k] <= 0.0:
                errors.append("%s has non-positive %s" % (s, k))
        if SEGMENTS[s]["node_p"] == SEGMENTS[s]["node_d"]:
            errors.append("%s is a self-loop" % s)

    for sub in sorted(SUBTREES):
        segs = _resolve_segments(sub)
        names = SUBTREES[sub]

        # every non-inlet proximal node must be fed by another segment
        distal = set(d["node_d"] for d in segs.values())
        for s in names:
            p = segs[s]["node_p"]
            if p != INLET_NODE and p not in distal:
                errors.append("%s: %s hangs off %s, which nothing feeds"
                              % (sub, s, p))

        # one parent per node
        seen = {}
        for s in names:
            nd = segs[s]["node_d"]
            if nd in seen:
                errors.append("%s: node %s is the distal end of both %s and %s"
                              % (sub, nd, seen[nd], s))
            seen[nd] = s

        # reachability from the aortic root
        reached = set([INLET_NODE])
        changed = True
        while changed:
            changed = False
            for s in names:
                if segs[s]["node_p"] in reached \
                        and segs[s]["node_d"] not in reached:
                    reached.add(segs[s]["node_d"])
                    changed = True
        for s in names:
            if segs[s]["node_d"] not in reached:
                errors.append("%s: %s unreachable from %s"
                              % (sub, s, INLET_NODE))

        terms = terminals_of(sub)
        for s in terms:
            if not segs[s]["node_d"].startswith("T_"):
                errors.append("%s: leaf node %s (segment %s) should be T_*"
                              % (sub, segs[s]["node_d"], s))

        # territory map: exists, keyed by the actual terminals, partitions 1..17
        if sub not in PERFUSION_REGIONS:
            errors.append("%s: no perfusion territory map" % sub)
        else:
            reg = PERFUSION_REGIONS[sub]
            extra = EXTRA_LV_TERMINALS.get(sub, {})
            for t in extra:
                if reg.get(t):
                    errors.append("%s: %s is listed as an extra-LV terminal "
                                  "but has AHA segments %s"
                                  % (sub, t, reg[t]))
            if sorted(reg.keys()) != sorted(terms):
                errors.append("%s: territory map covers %s but terminals are %s"
                              % (sub, sorted(reg.keys()), sorted(terms)))
            covered = sorted(sum([list(v) for v in reg.values()], []))
            if covered != list(range(1, 18)):
                errors.append("%s: territories are not a partition of AHA "
                              "1..17: %s" % (sub, covered))

        # terminal resistance: present and positive for every terminal
        if sub not in TERMINAL_RESISTANCE:
            errors.append("%s: no terminal resistance table" % sub)
        else:
            rt = TERMINAL_RESISTANCE[sub]
            for s in terms:
                if s not in rt:
                    errors.append("%s: no terminal resistance for %s"
                                  % (sub, s))
                elif rt[s] <= 0.0:
                    errors.append("%s: terminal resistance for %s is not "
                                  "positive" % (sub, s))

    if errors:
        raise AssertionError("coronary tables are malformed:\n  - "
                             + "\n  - ".join(errors))

    if verbose:
        for sub in sorted(SUBTREES):
            print("  %-14s %d segments, %d territories"
                  % (sub, len(SUBTREES[sub]), len(terminals_of(sub))))


check_tables()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class CoronaryRC(object):
    """Node-based R-C coronary network.

    Pressures are the states and flows are algebraic across resistances,
    matching MyoFE's circulation model.  The aortic root and the territory
    outlets are pinned; interior junctions and terminal nodes are solved.
    """

    def __init__(self, subtree, dt, terminal_resistance=None,
                 inertance=False):
        """subtree is a key of SUBTREES.  terminal_resistance may be None (no
        downstream resistance), a single number applied to every territory, or
        a dict keyed by terminal segment name; omitted, the calibrated table
        for this configuration is used."""

        self.subtree = subtree
        self.inertance = bool(inertance)
        self.segments = _resolve_segments(subtree)
        self.names = list(SUBTREES[subtree])
        self.dt = dt

        nodes = set()
        for s in self.names:
            nodes.add(self.segments[s]["node_p"])
            nodes.add(self.segments[s]["node_d"])

        upstream = set(self.segments[s]["node_p"] for s in self.names)
        self.terminal_nodes = sorted(n for n in nodes
                                     if n != INLET_NODE and n not in upstream)
        self.terminal_segment = dict(
            (self.segments[s]["node_d"], s) for s in self.names
            if self.segments[s]["node_d"] in self.terminal_nodes)
        self.terminals = [self.terminal_segment[n]
                          for n in self.terminal_nodes]

        self.regions = PERFUSION_REGIONS[subtree]

        if terminal_resistance is None:
            terminal_resistance = TERMINAL_RESISTANCE[subtree]

        if isinstance(terminal_resistance, dict):
            missing = [s for s in self.terminals
                       if s not in terminal_resistance]
            if missing:
                raise ValueError(
                    "terminal_resistance is missing entries for %s" % missing)
            self.Rt = dict((s, float(terminal_resistance[s]))
                           for s in self.terminals)
        else:
            self.Rt = dict((s, float(terminal_resistance))
                           for s in self.terminals)
        for s, v in self.Rt.items():
            if v <= 0.0:
                raise ValueError("terminal_resistance for %s must be "
                                 "positive, got %r" % (s, v))

        # with a terminal resistance the outlet node is no longer pinned:
        # IMP now sits beyond it, so the terminal node becomes an unknown
        self.free_nodes = sorted(n for n in nodes if n != INLET_NODE)
        self.idx = dict((n, i) for i, n in enumerate(self.free_nodes))
        self.n_p = len(self.free_nodes)

        if self.inertance:
            # segment flows are states too, appended after the pressures
            self.qidx = dict((s, self.n_p + i)
                             for i, s in enumerate(self.names))
            self.n = self.n_p + len(self.names)
        else:
            self.qidx = {}
            self.n = self.n_p

        self.C = dict((n, 0.0) for n in nodes)
        for s in self.names:
            self.C[self.segments[s]["node_d"]] += self.segments[s]["C"]

        self._assemble()

    def _pinned(self, node, P_AO, P_IMP):
        if node == INLET_NODE:
            return P_AO
        return P_IMP[self.terminal_segment[node]]

    def _assemble(self):
        if self.inertance:
            return self._assemble_rlc()
        A = np.zeros((self.n, self.n))
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] += self.C[node] / self.dt
        for s in self.names:
            a, b = self.segments[s]["node_p"], self.segments[s]["node_d"]
            g = 1.0 / self.segments[s]["R"]
            for u, v in ((a, b), (b, a)):
                if u in self.idx:
                    A[self.idx[u], self.idx[u]] += g
                    if v in self.idx:
                        A[self.idx[u], self.idx[v]] -= g
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            A[self.idx[nd], self.idx[nd]] += 1.0 / self.Rt[s]
        self.A = A
        if _HAVE_SCIPY and self.n > 0:
            self._lu = lu_factor(A)

    def _assemble_rlc(self):
        """Pressures and flows together.  Rows 0..n_p-1 are node mass balance,
        rows n_p.. are the segment momentum equations."""
        A = np.zeros((self.n, self.n))
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] += self.C[node] / self.dt
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            A[self.idx[nd], self.idx[nd]] += 1.0 / self.Rt[s]
        for s in self.names:
            j = self.qidx[s]
            a, d = self.segments[s]["node_p"], self.segments[s]["node_d"]
            if a in self.idx:
                A[self.idx[a], j] += 1.0        # leaves the proximal node
                A[j, self.idx[a]] -= 1.0
            if d in self.idx:
                A[self.idx[d], j] -= 1.0        # enters the distal node
                A[j, self.idx[d]] += 1.0
            A[j, j] += self.segments[s]["L"] / self.dt + self.segments[s]["R"]
        self.A = A
        if _HAVE_SCIPY and self.n > 0:
            self._lu = lu_factor(A)

    def _bc_load_rlc(self, P_AO, P_IMP):
        b = np.zeros(self.n)
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            b[self.idx[nd]] += P_IMP[s] / self.Rt[s]
        for s in self.names:
            j = self.qidx[s]
            if self.segments[s]["node_p"] == INLET_NODE:
                b[j] += P_AO
            if self.segments[s]["node_d"] == INLET_NODE:
                b[j] -= P_AO
        return b

    def _bc_load(self, P_AO, P_IMP):
        if self.inertance:
            return self._bc_load_rlc(P_AO, P_IMP)
        b = np.zeros(self.n)
        for s in self.names:
            a, d = self.segments[s]["node_p"], self.segments[s]["node_d"]
            g = 1.0 / self.segments[s]["R"]
            for u, v in ((a, d), (d, a)):
                if u in self.idx and v not in self.idx:
                    b[self.idx[u]] += g * self._pinned(v, P_AO, P_IMP)
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
        if self.inertance:
            for s in self.names:
                j = self.qidx[s]
                b[j] += self.segments[s]["L"] / self.dt * P_prev[j]
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
        if self.inertance:
            for s in self.names:
                j = self.qidx[s]
                A[j, j] -= self.segments[s]["L"] / self.dt
        return np.linalg.solve(A, self._bc_load(P_AO, P_IMP))

    def pressure(self, P, node, P_AO, P_IMP):
        if node in self.idx:
            return P[self.idx[node]]
        return self._pinned(node, P_AO, P_IMP)

    def flows(self, P, P_AO, P_IMP):
        """Flow through each epicardial segment, positive proximal to distal.
        For a conduit such as LMCA this is the sum of its children."""
        if self.inertance:
            return dict((s, P[self.qidx[s]]) for s in self.names)
        q = {}
        for s in self.names:
            pa = self.pressure(P, self.segments[s]["node_p"], P_AO, P_IMP)
            pd = self.pressure(P, self.segments[s]["node_d"], P_AO, P_IMP)
            q[s] = (pa - pd) / self.segments[s]["R"]
        return q

    def perfusion(self, P, P_AO, P_IMP):
        """Flow delivered into each territory, across the terminal
        resistance.  This is the quantity Q that feeds Eq. 19."""
        out = {}
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            out[s] = (P[self.idx[nd]] - P_IMP[s]) / self.Rt[s]
        return out

    def describe(self):
        print("  config    : %s" % self.subtree)
        print("  segments  : %d  %s" % (len(self.names), ", ".join(self.names)))
        print("  free nodes: %d  %s" % (self.n, ", ".join(self.free_nodes)))
        print("  pinned    : %-8s = P_AO" % INLET_NODE)
        for s in self.terminals:
            print("              %-8s = P_IMP[%s]  (beyond Rt = %.4g)"
                  % ("IMP_" + s, s, self.Rt[s]))
        for s in self.terminals:
            print("  %-5s -> AHA %s" % (s, self.regions[s]))


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("Coronary segment table -- Wang et al. Table 1")
    print("=" * 70)
    print("%-6s %9s %9s %10s   %-8s -> %-8s"
          % ("seg", "R", "L", "C", "prox", "dist"))
    print("-" * 70)
    for s in SEGMENT_ORDER:
        d = SEGMENTS[s]
        print("%-6s %9.4f %9.4f %10.5f   %-8s -> %-8s"
              % (s, d["R"], d["L"], d["C"], d["node_p"], d["node_d"]))

    print()
    print("Configurations")
    print("=" * 70)
    check_tables(verbose=True)

    print()
    print("Production configuration")
    print("=" * 70)
    m = CoronaryRC("lad_lcx_rca", 1.0e-3)
    m.describe()

    print()
    print("Steady state, P_AO = 86.0 mmHg, IMP = 0")
    print("=" * 70)
    P_AO = 86.0
    P_IMP = dict((s, 0.0) for s in m.terminals)
    P = m.steady_state(P_AO, P_IMP)
    q = m.perfusion(P, P_AO, P_IMP)
    total = 0.0
    for s in m.terminals:
        print("  %-5s %8.3f mL/s = %6.1f mL/min" % (s, q[s], q[s] * 60.0))
        total += q[s]
    print("  total %8.3f mL/s = %6.1f mL/min" % (total, total * 60.0))

    print()
    print("Steal check -- systolic peak, IMP 70/70/50 mmHg")
    print("=" * 70)
    P_IMP = {"LAD": 70.0, "LCX": 70.0, "RCA": 50.0}
    P = m.steady_state(P_AO, P_IMP)
    q = m.perfusion(P, P_AO, P_IMP)
    for s in m.terminals:
        print("  %-5s %8.3f mL/s   %s"
              % (s, q[s], "forward" if q[s] > 0 else "REVERSED -- steal"))
    if min(q.values()) <= 0.0:
        raise AssertionError("flow reverses under peak systolic compression")
    print("  all territories forward-flowing at peak compression.")
