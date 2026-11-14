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

TERRITORY OUTFLOW (JSON: "outflow")
-----------------------------------
How blood leaves each terminal node T_i into its territory, with P_T the
node pressure and IMP_i the territory's intramyocardial pressure:

  "linear"            Q_i = (P_T - IMP_i) / R_i                  (default)
      IMP acts as a reservoir pressure beyond R_i.  Q_i goes negative
      whenever IMP_i > P_T, and stays negative on average once a fully
      dilated bed sits at P_T ~ mean IMP (seen at LAD 10% and 1%: blood
      pushed out of LAD3/LAD4 and into LAD1).

  "diode"             Q_i = max(P_T - IMP_i, 0) / R_i
      The vascular waterfall (Downey & Kirk, Circ Res 1975): when IMP
      exceeds the pressure inside them the small vessels collapse and flow
      STOPS; it cannot reverse, because nothing on the venous side can push
      blood back.  This is the ideal diode / ramp function of Garcia et al.,
      J Appl Physiol 2009, Eq. 3.  With nothing else squeezed by IMP, the
      brief early-systolic backflow disappears too.

  "diode_compliance"  Q_i = max(P_T - IMP_i, 0) / R_i
                            + C_im,i d(P_T - IMP_i)/dt
      The diode plus an intramyocardial compliance squeezed by IMP -- the
      full form of Garcia Eq. 3 (the "intramyocardial pump", Spaan et al.,
      Circ Res 1981).  When IMP rises in early systole it pushes blood out
      of C_im back into the artery: brief backflow, physiological.  Over a
      periodic beat the C_im term integrates to zero, so the beat-mean flow
      is the diode's, never negative.  C_im,i: Garcia's whole-LAD value,
      4e-4 mL/mmHg, shared in proportion to resting flow (equivalently mass,
      at 1 mL/min/g), so every territory has the same compliance per gram;
      see intramyocardial_compliance().

The diode makes each step piecewise linear.  It is solved by an active set:
each terminal is "open" (P_T > IMP, conductance 1/R_i) or "closed" (no
resistive outflow); solve, flip any terminal whose state contradicts its
pressures, re-solve.  The matrix for each open/closed pattern is factorized
once and cached, so a step is still one or two back-substitutions.  "linear"
skips all of this and is bit-for-bit the earlier solver.

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

OUTFLOWS = ("linear", "diode", "diode_compliance")

# Garcia et al. 2009, Appendix B: total intramyocardial compliance of the
# LAD bed, fitted to LAD flow waveforms (mL/mmHg).
GARCIA_C_LAD = 4.0e-4
LAD_TERMINALS = ("LAD1", "LAD3", "LAD4")


def intramyocardial_compliance(q_rest, c_lad=GARCIA_C_LAD,
                               lad_terminals=LAD_TERMINALS):
    """C_im per terminal: c_lad shared over the LAD territories in
    proportion to resting flow, and the same compliance per unit resting
    flow given to every other terminal.  q_rest: {terminal: resting flow},
    any unit.  Returns {terminal: mL/mmHg}."""
    lad = sum(float(q_rest[s]) for s in lad_terminals if s in q_rest)
    if not lad > 0.0:
        raise ValueError("intramyocardial_compliance: no resting flow for the "
                         "LAD territories %s" % (lad_terminals,))
    per_flow = float(c_lad) / lad
    return dict((s, per_flow * float(q)) for s, q in q_rest.items())


def steady_outflow(outflow):
    """The outflow to use for a STEADY solve: a compliance carries no mean
    flow, so "diode_compliance" is just "diode" at steady state."""
    return "diode" if outflow == "diode_compliance" else outflow


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
    # There is deliberately no "full" entry.  The full tree's territory map is
    # dependencies/aha_segmentation.PERFUSION_REGIONS -- the one validated in
    # ParaView -- and perfusion.py reads it from there.  See
    # TERRITORIES_DEFINED_IN_AHA below.
}

# Configurations whose territory map lives in dependencies/aha_segmentation.py
# rather than here.  Keeping a copy would let the two drift apart silently, so
# there is exactly one.  aha_segmentation checks its own partition of AHA 1..17
# at import, so nothing is lost by not checking it again here.
TERRITORIES_DEFINED_IN_AHA = ("full",)


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

# ---------------------------------------------------------------------------
# Physiological terminal resistances
# ---------------------------------------------------------------------------
# Wang's published Z give LV perfusion of 2.19 mL/min/g on this mesh -- about
# twice the healthy resting value -- and very uneven: LAD4 received 6.5
# mL/min/g while LAD1 received 1.2.  That is not a heart-size mismatch (Wang
# report EDV 100.9 mL against 90.6 here); it is their parameters.
#
# These values give every LV territory the same resting perfusion,
# 1.0 mL/min/g, the healthy-adult PET median (Lassen et al., Eur J Nucl Med
# Mol Imaging 2026, doi:10.1007/s00259-026-07983-3: Rb-82 PET in 277 healthy
# adults, median 1.00, IQR 0.82-1.18 mL/g/min).  Uniform per gram is a
# first-order approximation: at territory scale (8-27 g) the fractal law of
# Bassingthwaighte et al. predicts only ~10% inter-territory variation, far
# below the 75% the published Z produced.
#
# Calibrated against the current healthy baseline ("run B"): the
# labmate-derived parameter set (baseline_guccioneMSB -- circulation, passive
# law, gentler baroreflex) with k_1 = 5.5 and k_act = 0.0830.  EDV 113.7 mL,
# ESV 54.3 mL, EF 52.2%, HR 67.6 bpm, peak LV pressure 114.8 mmHg -- volumes
# and heart rate inside the clinical-control ranges.  Chosen over the stronger
# k_1 6.0 / k_act 0.090 set because it keeps the better coronary
# diastolic:systolic flow ratio (1.88 against 1.74 into the muscle), as
# agreed with Dr. Wenk.  Baroreflex from 15 s, run to 50 s; P_AO and stress
# IMP recorded over t = 40.2-49.9 s (11 whole beats; baroreflex settled,
# drift 0.015 mmHg/s), R-L-C, iterating Z_i <- Z_i * Q_i/Q_target_i to
# 0.001%.  Because the coupling is one-way, the mechanics do not depend on Z,
# so the calibration is exact for that operating point.  Recalibrate whenever
# the heart's parameters change.
#
# Other calibrations, for reference:
#   k_1 6.0, k_act 0.090 ("run C", data_6):
#     LAD1 163.23  LAD3 276.53  LAD4 528.89  MARG1 258.90  MARG2 247.79
#     MARG3 337.89  LCX3 245.60  PDA 255.91  PLA 126.41
#   labmate set, k_1 5.15, k_act 0.0806 (data_4):
#     LAD1 157.10  LAD3 270.12  LAD4 511.98  MARG1 249.31  MARG2 238.52
#     MARG3 329.20  LCX3 236.56  PDA 247.19  PLA 122.14
#   original parameter set (data_3):
#     LAD1 176.95  LAD3 289.95  LAD4 567.33  MARG1 278.29  MARG2 268.36
#     MARG3 357.59  LCX3 266.09  PDA 275.49  PLA 135.77
#
# Territory masses from the mesh (reference volume x 1.05 g/mL):
#   LAD1 27.33  LAD3 15.89  LAD4 8.47  MARG1 17.07  MARG2 17.89
#   MARG3 13.26  LCX3 17.89  PDA 17.05        LV total 134.8 g
#
# PLA has no LV tissue.  It takes the same factor the LV total needed,
# giving 40.1 mL/min for the right-ventricular region.
#
# "rt_set": ["physiological"] (the default) uses these; "published" uses
# Wang's Table 1 values.

TERMINAL_RESISTANCE_PHYSIO = {
    "full": {
        "LAD1":   159.32,
        "LAD3":   272.52,
        "LAD4":   517.92,
        "MARG1":  252.78,
        "MARG2":  241.75,
        "MARG3":  332.68,
        "LCX3":   239.69,
        "PDA":    250.24,
        "PLA":    123.67,
    },
}


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


def apply_stenosis(segments, stenosis, check_units=True):
    """Narrow segments in place, following Wang et al. Eq. 1.

    stenosis : {segment name: percent AREA reduction}, e.g. {"LMCA": 80}.
               80 means 80% of the lumen area is lost, beta = A_s/A_0 = 0.2.

        R_s = R beta^-2     L_s = L beta^-1     C_s = C beta^(3/2)

    These follow from R ~ D^-4, L ~ D^-2 and C ~ D^3 (Wang Eq. 2) with the
    segment length fixed, since D_s/D_0 = beta^(1/2).  segments must be the
    tree's own copies (as _resolve_segments returns), so SEGMENTS -- Wang's
    Table 1 -- is never changed.

    Returns {name: dict(percent, beta, before=(R, L, C), after=(R, L, C))}
    for logging.
    """
    applied = {}
    for name in sorted(stenosis):
        if name not in segments:
            raise ValueError(
                "stenosis on '%s': no such segment in this tree.  Segments "
                "are %s" % (name, sorted(segments)))
        pct = float(stenosis[name])
        if not 0.0 <= pct < 100.0:
            raise ValueError(
                "stenosis on %s is %g%%: give percent area reduction, "
                "0 <= value < 100" % (name, pct))
        if check_units and 0.0 < pct < 1.0:
            raise ValueError(
                "stenosis on %s is %g: give PERCENT area reduction, e.g. 80 "
                "for 80%%, not 0.8" % (name, pct))
        beta = 1.0 - pct / 100.0
        seg = segments[name]
        before = (seg["R"], seg["L"], seg["C"])
        seg["R"] = seg["R"] * beta ** -2
        seg["L"] = seg["L"] * beta ** -1
        seg["C"] = seg["C"] * beta ** 1.5
        applied[name] = dict(percent=pct, beta=beta, before=before,
                             after=(seg["R"], seg["L"], seg["C"]))
    return applied


# Beat-mean aortic pressure and IMP (mmHg) of the healthy run B baseline,
# data_8.csv, t = 65.9-79.2 s.  The tree is linear, so its beat-mean flows
# are exactly its steady solution at these mean pressures (checked: every
# territory within 0.01% of the run's own mean flow).  Used only to turn
# "perfusion_percent" into a narrowing.  Re-measure with Q_ref if the heart
# or the terminal resistances change.
REF_MEAN_P_AO = 90.688
REF_MEAN_IMP = {
    "LAD1": 16.934, "LAD3": 15.756, "LAD4": 15.131,
    "MARG1": 17.338, "MARG2": 16.533, "MARG3": 14.808,
    "LCX3": 16.601, "PDA": 17.224, "PLA": 5.488,
}


# ---------------------------------------------------------------------------
# Coronary autoregulation
# ---------------------------------------------------------------------------
# The microvessels of each territory -- its terminal resistance R_i -- adjust
# to hold that territory's flow at its target Q_target,i, within limits.  Once
# per beat, with Qbar_i the beat-mean flow:
#
#     R*_i = R_i * Qbar_i / Q_target,i        the resistance that would have
#                                             given the target this beat
#     R_i  <- R_i + (R*_i - R_i) / n_ar       move 1/n_ar of the way there
#     R_i  <- clip(R_i, R_min,i, R_max,i)
#
#     R_min,i = R0_i / CFR                    fully dilated: CFR is the ratio
#                                             of hyperemic to resting flow at
#                                             the same perfusion pressure,
#                                             = R0/R_min (Garcia et al. 2009,
#                                             J Appl Physiol, Eqs. 7-8)
#     R_max,i = R0_i (P_upper - IMP_i) / (P_AO,ref - IMP_i)
#                                             fully constricted: flow held up
#                                             to P_upper = 120 mmHg (Goodwill
#                                             et al. 2017, Compr Physiol)
#
# Between the limits flow returns to target; at R_min the territory is
# maximally dilated and flow follows pressure passively.  The lower limit of
# the range follows from the bounds:
#
#     P_low,i = IMP_i + (P_AO,ref - IMP_i) / CFR
#
# which with CFR = 3.3 is 38-40 mmHg for every LV territory (about 40 mmHg in
# conscious dogs: Canty 1988; Duncker et al. 2015, Fig. 1).  n_ar = 5 beats
# gives a half-response of 4.5-5.0 s after a 20-30 mmHg drop in aortic
# pressure at 68 bpm, measured by replaying a recorded beat through
# perfusion.py (measured in goats: 4.9 s, Dankelman et al. 1989, J Physiol
# 419:703).  In continuous form the law is d(ln R)/dt = (Q/Q_target - 1)/tau,
# tau = n_ar beats: integral control, so flow returns exactly to target.  The
# same multiplicative update is how TERMINAL_RESISTANCE_PHYSIO was calibrated.

AUTOREG_P_UPPER = 120.0      # mmHg


def autoregulation_bounds(R0, cfr, p_upper=AUTOREG_P_UPPER, P_AO=None,
                          P_IMP=None):
    """R_min and R_max for every terminal (dicts keyed like R0)."""
    if P_AO is None:
        P_AO = REF_MEAN_P_AO
    if P_IMP is None:
        P_IMP = REF_MEAN_IMP
    cfr = float(cfr)
    if cfr <= 1.0:
        raise ValueError("autoregulation: cfr = %g; the flow reserve must be "
                         "above 1 (e.g. 3.3)" % cfr)
    if float(p_upper) <= P_AO:
        raise ValueError("autoregulation: p_upper = %g mmHg must be above the "
                         "reference aortic pressure %.1f" % (p_upper, P_AO))
    r_min, r_max = {}, {}
    for s, r in R0.items():
        if s not in P_IMP:
            raise ValueError("autoregulation: no reference IMP for %s; it is "
                             "only set up for the full tree" % s)
        r_min[s] = r / cfr
        r_max[s] = r * (float(p_upper) - P_IMP[s]) / (P_AO - P_IMP[s])
    return r_min, r_max


def autoregulation_update(R, qbar, q_target, r_min, r_max, n_ar=1.0):
    """One step of the law above for every terminal; returns the new R.
    A beat-mean flow at or below zero (no forward flow) asks for full
    dilation, which the clip turns into R_min."""
    out = {}
    for s in R:
        r_star = R[s] * qbar[s] / q_target[s]
        r = R[s] + (r_star - R[s]) / float(n_ar)
        out[s] = min(max(r, r_min[s]), r_max[s])
    return out


def autoregulated_state(tree, q_target, r_min, r_max, P_AO, P_IMP,
                        R_start=None, tol=1e-10, max_iter=20000):
    """Where autoregulation settles at constant pressures: the fixed point of
    the update.  Sets tree's terminal resistances to it and returns them.

    The full step (n_ar = 1) is used -- the fixed point does not depend on
    n_ar.  It converges because each terminal sees the rest of the tree as a
    resistance R_up in series: R <- R Q/Q_target contracts with factor
    R_up/(R + R_up) < 1."""
    R = dict(R_start) if R_start else dict(tree.Rt)
    for it in range(max_iter):
        tree.set_terminal_resistance(R)
        P = tree.steady_state(P_AO, P_IMP)
        q = tree.perfusion(P, P_AO, P_IMP)
        new = autoregulation_update(R, q, q_target, r_min, r_max)
        change = max(abs(new[s] / R[s] - 1.0) for s in R)
        R = new
        if change < tol:
            tree.set_terminal_resistance(R)
            return R
    raise RuntimeError("autoregulation steady state did not converge "
                       "(last relative change %.2e)" % change)


def narrowing_for_flow(subtree, percent_flow, terminal_resistance=None,
                       inertance=False, fixed_stenosis=None,
                       P_AO=None, P_IMP=None, autoregulation=None,
                       outflow="linear"):
    """Turn {segment: percent of healthy flow} into {segment: percent AREA
    reduction}, the input apply_stenosis takes.

    "LAD": 20 means the LAD carries 20% of the flow it carries in the
    healthy heart.  The narrowing is found once, against the healthy heart
    (REF_MEAN_P_AO, REF_MEAN_IMP): it is a fixed narrowing, so once the
    simulation runs, flow still answers to the heart -- that is the feedback
    under study.  Several targets interact through shared vessels (LAD and
    LCX both hang off LMCA), so they are solved together.

    autoregulation=dict(q_target=, r_min=, r_max=) (mL/s and terminal
    resistances, keyed by terminal): the flows are those AFTER the
    microvessels have adapted, so "LAD": 20 means 20% even once the LAD bed
    has dilated as far as it can.  At rest any lasting loss of flow means the
    bed is already maximally dilated, so the narrowing comes out tighter than
    without autoregulation.

    outflow: the run's territory outflow (see TERRITORY OUTFLOW); solved
    here at steady state, where "diode_compliance" acts as "diode".
    """
    fixed = dict(fixed_stenosis or {})
    ss_outflow = steady_outflow(outflow)
    if P_AO is None:
        P_AO = REF_MEAN_P_AO
    if P_IMP is None:
        P_IMP = REF_MEAN_IMP
    warm = {}                    # last autoregulated R, to start the next solve

    def flows(area):
        sten = dict(fixed)
        sten.update(dict((s, a) for s, a in area.items() if a > 0.0))
        tree = CoronaryRC(subtree, 1.0, terminal_resistance=terminal_resistance,
                          inertance=inertance, stenosis=sten, check_units=False,
                          outflow=ss_outflow)
        missing = [t for t in tree.terminals if t not in P_IMP]
        if missing:
            raise ValueError("perfusion_percent: no reference IMP for %s; it "
                             "is only set up for the full tree" % missing)
        if autoregulation is not None:
            warm['R'] = autoregulated_state(
                tree, autoregulation['q_target'], autoregulation['r_min'],
                autoregulation['r_max'], P_AO, P_IMP, R_start=warm.get('R'))
        P = tree.steady_state(P_AO, P_IMP)
        return tree.flows(P, P_AO, P_IMP)

    apply_stenosis(_resolve_segments(subtree), fixed)   # check the user's own
    names = flows({}).keys()
    target = {}
    for s, v in percent_flow.items():
        if s not in names:
            raise ValueError("perfusion_percent on '%s': no such vessel.  "
                             "Vessels are %s" % (s, sorted(names)))
        if s in fixed:
            raise ValueError("'%s' is in both stenosis and perfusion_percent; "
                             "give it in one only" % s)
        v = float(v)
        if not 0.0 < v <= 100.0:
            raise ValueError("perfusion_percent on %s is %g: give the percent "
                             "of normal flow, 0 < value <= 100 (100 = healthy; "
                             "use e.g. 1 for almost none)" % (s, v))
        if v < 1.0:
            raise ValueError("perfusion_percent on %s is %g: give a PERCENT, "
                             "e.g. 20 for 20%%, not 0.2" % (s, v))
        target[s] = v / 100.0

    q0 = flows({})                       # healthy: no narrowing anywhere
    area = dict((s, 0.0) for s in target)
    for sweep in range(200):
        before = dict(area)
        for s in sorted(target):
            if target[s] >= 1.0:
                area[s] = 0.0
                continue
            # flow through s rises with its open fraction beta; bisect log(beta)
            lo, hi = np.log(1e-8), 0.0
            for it in range(80):
                mid = 0.5 * (lo + hi)
                area[s] = 100.0 * (1.0 - np.exp(mid))
                if flows(area)[s] / q0[s] > target[s]:
                    hi = mid
                else:
                    lo = mid
            area[s] = 100.0 * (1.0 - np.exp(0.5 * (lo + hi)))
        q = flows(area)
        if all(abs(q[s] / q0[s] - target[s]) < 1e-6 for s in target):
            return area
        if all(abs(area[s] - before[s]) < 1e-9 for s in target):
            break                        # stuck: the targets contradict each other
    raise ValueError(
        "perfusion_percent %s cannot all hold at once: the closest is %s.  "
        "A vessel downstream of another already limits its flow (e.g. LAD3 at "
        "20%% caps LAD2), so a parent cannot get MORE than its children allow."
        % (dict((s, 100.0 * target[s]) for s in sorted(target)),
           dict((s, round(100.0 * q[s] / q0[s], 1)) for s in sorted(target))))


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
        if sub in TERRITORIES_DEFINED_IN_AHA:
            pass    # checked by aha_segmentation itself, and by perfusion.py
        elif sub not in PERFUSION_REGIONS:
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
                 inertance=False, stenosis=None, check_units=True,
                 outflow="linear", c_im=None):
        """subtree is a key of SUBTREES.  terminal_resistance may be None (no
        downstream resistance), a single number applied to every territory, or
        a dict keyed by terminal segment name; omitted, the calibrated table
        for this configuration is used.

        outflow is one of OUTFLOWS (see TERRITORY OUTFLOW).  c_im, required
        for "diode_compliance", is {terminal: mL/mmHg}."""

        if outflow not in OUTFLOWS:
            raise ValueError("outflow must be one of %s, got '%s'"
                             % (list(OUTFLOWS), outflow))
        self.outflow = outflow
        self.subtree = subtree
        self.inertance = bool(inertance)
        self.segments = _resolve_segments(subtree)
        # Narrowed segments, if any: {name: percent area reduction}.
        # check_units=False only for narrowings the code computed itself
        # (perfusion_percent), which can legitimately be below 1%.
        self.stenosis = apply_stenosis(self.segments, stenosis or {},
                                       check_units=check_units)
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

        # Empty for the full tree, whose map is in aha_segmentation.py.
        self.regions = PERFUSION_REGIONS.get(subtree, {})

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

        # Territory outflow.  open[s]: resistive outflow active (always True
        # for "linear").  c_im[s]: compliance squeezed by IMP (zero unless
        # "diode_compliance").
        self.open = dict((s, True) for s in self.terminals)
        self.c_im = dict((s, 0.0) for s in self.terminals)
        if outflow == "diode_compliance":
            if c_im is None:
                raise ValueError("outflow 'diode_compliance' needs c_im, the "
                                 "intramyocardial compliance per terminal")
            missing = [s for s in self.terminals if s not in c_im]
            if missing:
                raise ValueError("c_im is missing entries for %s" % missing)
            for s in self.terminals:
                v = float(c_im[s])
                if v < 0.0:
                    raise ValueError("c_im for %s must be >= 0, got %r"
                                     % (s, v))
                self.c_im[s] = v
        self._imp_prev = None        # IMP at the previous step (C_im term)
        self._last_qc = dict((s, 0.0) for s in self.terminals)
        self._lu_cache = {}

        self._assemble()

    def set_terminal_resistance(self, Rt):
        """Change terminal resistances mid-run (autoregulation, once per beat)
        and refactorize.  Rt is a dict keyed by terminal; terminals left out
        keep their value.  The state vector lives with the caller, so it
        carries over unchanged."""
        for s, v in Rt.items():
            if s not in self.Rt:
                raise ValueError("no terminal '%s'; terminals are %s"
                                 % (s, self.terminals))
            v = float(v)
            if not v > 0.0:
                raise ValueError("terminal_resistance for %s must be "
                                 "positive, got %r" % (s, v))
            self.Rt[s] = v
        self._lu_cache = {}
        self._assemble()

    def forget_imp_history(self):
        """Make the next step() treat its IMP as unchanged from the step
        before, so a C_im sees no artificial jump when stepping starts from
        a state solved at a different IMP (e.g. the prescribed IMP at t=0)."""
        self._imp_prev = None
        self._last_qc = dict((s, 0.0) for s in self.terminals)

    def _pinned(self, node, P_AO, P_IMP):
        if node == INLET_NODE:
            return P_AO
        return P_IMP[self.terminal_segment[node]]

    def _open_key(self):
        return tuple(self.open[s] for s in self.terminals)

    def _assemble(self):
        """Matrix and LU for the current open/closed pattern, from the cache
        when this pattern has been seen since the last resistance change."""
        key = self._open_key()
        if key not in self._lu_cache:
            A = self._build_rlc() if self.inertance else self._build()
            lu = lu_factor(A) if (_HAVE_SCIPY and self.n > 0) else None
            self._lu_cache[key] = (A, lu)
        self.A, self._lu = self._lu_cache[key]

    def _build(self):
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
            if self.open[s]:
                A[self.idx[nd], self.idx[nd]] += 1.0 / self.Rt[s]
        self._add_c_im(A)
        return A

    def _build_rlc(self):
        """Pressures and flows together.  Rows 0..n_p-1 are node mass balance,
        rows n_p.. are the segment momentum equations."""
        A = np.zeros((self.n, self.n))
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] += self.C[node] / self.dt
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            if self.open[s]:
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
        self._add_c_im(A)
        return A

    def _add_c_im(self, A):
        # C_im d(P_T - IMP)/dt leaves the terminal node: C_im/dt on its
        # diagonal; the old P_T and the change in IMP go to the load.
        if self.outflow != "diode_compliance":
            return
        for nd in self.terminal_nodes:
            i = self.idx[nd]
            A[i, i] += self.c_im[self.terminal_segment[nd]] / self.dt

    def _bc_load_rlc(self, P_AO, P_IMP):
        b = np.zeros(self.n)
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            if self.open[s]:
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
            if self.open[s]:
                b[self.idx[nd]] += P_IMP[s] / self.Rt[s]
        return b

    def _update_open(self, P, P_IMP):
        """Active-set check for the diode.  Flips every terminal whose state
        contradicts the solution just found; returns True if any flipped."""
        changed = False
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            dp = P[self.idx[nd]] - P_IMP[s]
            if self.open[s] and dp < 0.0:
                self.open[s] = False
                changed = True
            elif not self.open[s] and dp > 0.0:
                self.open[s] = True
                changed = True
        return changed

    def _solve_step(self, P_prev, P_AO, P_IMP):
        b = self._bc_load(P_AO, P_IMP)
        for node in self.free_nodes:
            i = self.idx[node]
            b[i] += self.C[node] / self.dt * P_prev[i]
        if self.inertance:
            for s in self.names:
                j = self.qidx[s]
                b[j] += self.segments[s]["L"] / self.dt * P_prev[j]
        if self.outflow == "diode_compliance":
            imp_prev = self._imp_prev if self._imp_prev is not None else P_IMP
            for nd in self.terminal_nodes:
                s = self.terminal_segment[nd]
                i = self.idx[nd]
                b[i] += self.c_im[s] / self.dt * (P_prev[i] - imp_prev[s]
                                                  + P_IMP[s])
        if _HAVE_SCIPY:
            return lu_solve(self._lu, b)
        return np.linalg.solve(self.A, b)

    def step(self, P_prev, P_AO, P_IMP):
        if self.n == 0:
            return np.zeros(0)
        if self.outflow == "linear":
            return self._solve_step(P_prev, P_AO, P_IMP)

        # Diode: start from the last step's open/closed pattern (it rarely
        # changes within one step) and re-solve until consistent.  The
        # network is a monotone (M-matrix) system, so this settles in a few
        # passes; a pattern seen once is factorized once.
        for it in range(2 * len(self.terminals) + 2):
            P = self._solve_step(P_prev, P_AO, P_IMP)
            if not self._update_open(P, P_IMP):
                break
            self._assemble()
        if self.outflow == "diode_compliance":
            imp_prev = self._imp_prev if self._imp_prev is not None else P_IMP
            for nd in self.terminal_nodes:
                s = self.terminal_segment[nd]
                i = self.idx[nd]
                self._last_qc[s] = self.c_im[s] / self.dt * (
                    (P[i] - P_IMP[s]) - (P_prev[i] - imp_prev[s]))
            self._imp_prev = dict(P_IMP)
        return P

    def _steady_solve(self, P_AO, P_IMP):
        A = self.A.copy()
        for node in self.free_nodes:
            i = self.idx[node]
            A[i, i] -= self.C[node] / self.dt
        if self.inertance:
            for s in self.names:
                j = self.qidx[s]
                A[j, j] -= self.segments[s]["L"] / self.dt
        if self.outflow == "diode_compliance":
            for nd in self.terminal_nodes:
                i = self.idx[nd]
                A[i, i] -= self.c_im[self.terminal_segment[nd]] / self.dt
        return np.linalg.solve(A, self._bc_load(P_AO, P_IMP))

    def steady_state(self, P_AO, P_IMP):
        """Constant pressures: no compliance or inertance terms.  With a
        diode, terminals whose pressure stays below IMP are closed."""
        if self.n == 0:
            return np.zeros(0)
        if self.outflow == "linear":
            return self._steady_solve(P_AO, P_IMP)
        for it in range(2 * len(self.terminals) + 2):
            P = self._steady_solve(P_AO, P_IMP)
            if not self._update_open(P, P_IMP):
                break
            self._assemble()
        self._last_qc = dict((s, 0.0) for s in self.terminals)
        self._imp_prev = dict(P_IMP)
        return P

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
        resistance.  This is the quantity Q that feeds Eq. 19.

        "diode": the resistive flow, never negative.  "diode_compliance":
        plus the flow into C_im, from the most recent step() (zero after
        steady_state()), so call it right after the solve that produced P."""
        out = {}
        for nd in self.terminal_nodes:
            s = self.terminal_segment[nd]
            dp = P[self.idx[nd]] - P_IMP[s]
            if self.outflow == "linear":
                out[s] = dp / self.Rt[s]
            else:
                q = max(dp, 0.0) / self.Rt[s] if self.open[s] else 0.0
                out[s] = q + self._last_qc[s]
        return out

    def describe(self):
        print("  config    : %s" % self.subtree)
        print("  segments  : %d  %s" % (len(self.names), ", ".join(self.names)))
        print("  free nodes: %d  %s" % (self.n, ", ".join(self.free_nodes)))
        print("  outflow   : %s" % self.outflow)
        print("  pinned    : %-8s = P_AO" % INLET_NODE)
        for s in self.terminals:
            print("              %-8s = P_IMP[%s]  (beyond Rt = %.4g)"
                  % ("IMP_" + s, s, self.Rt[s]))
        for s in self.terminals:
            print("  %-5s -> AHA %s"
                  % (s, self.regions.get(s, "see aha_segmentation.py")))


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
