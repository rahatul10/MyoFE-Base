# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 2026

@author: Islam, Rahatul

AHA 17-SEGMENT BULL'S-EYE SEGMENTATION FOR THE LEFT VENTRICLE
=============================================================

PURPOSE
-------
Assign every point in the LV wall an AHA segment id from 1 to 17, based only
on its position.  No manual Gauss-point picking, no mesh-specific index lists.
The label is a *function of position*, so it is mesh-independent and survives
mesh refinement.

WHAT THIS IS AND IS NOT
-----------------------
The AHA 17-segment model (Cerqueira et al., Circulation 2002) is a *display
and reporting convention* from cardiac imaging.  It is NOT a structure that can
be "discovered" from the geometry.  In particular, the circumferential origin
(theta = 0) is anchored, in imaging, to the anterior RV insertion point.  This
LV-only mesh has no RV, and an idealised truncated ellipsoid is rotationally
symmetric about its long axis -- so there is NO direction in the geometry that
is intrinsically "anterior" or "septal".  We therefore DECLARE theta = 0 via
the c_ref parameter.  Calling segment 8 "anteroseptal" is a naming convention
we impose, not anatomy we recover.  This must be stated explicitly in any paper.

THE PIPELINE
------------
    P (x,y,z)
      -> lambda  : normalised apex-to-base coordinate, 0 = apex, 1 = base
      -> theta   : circumferential angle about the long axis
      -> longitudinal band  (apex cap / apical / mid / basal)
      -> circumferential sector  (4 sectors apically, 6 sectors mid & basal)
      -> AHA segment id (1..17)

Segment 17 (the apex cap) uses lambda ONLY -- no angle.  Near the apex the
circumferential direction degenerates (every direction converges), so theta is
numerically unstable and anatomically meaningless there.

MESH-SPECIFIC GEOMETRY (measured, not guessed)
----------------------------------------------
Measured from  demos/HCM_paper/simulations_baseline_final/
                   baseline_final_Kappa_4_k1_3.7/sim_inputs/ellipsoidal.hdf5

    base plane        : z = 0 exactly (planar disc, centroid on the axis)
    epicardial apex   : z = -0.7315
    endocardial apex  : z = -0.6655
    long axis         : +z  (apex -> base)
    axis length       : 0.7315

    => lambda = (z + 0.7315) / 0.7315

    lambda_c = 0.0902   the ENDOCARDIAL apex, i.e. where the cavity closes.
                        This is the anatomically principled definition of the
                        apex cap: segment 17 is the myocardium DISTAL to the LV
                        cavity.  It is NOT an arbitrary cutoff like 0.05 (which
                        would have been wrong by nearly a factor of two here).

    lambda_A = 0.3935   \  the cavity-bearing region (lambda_c .. 1) split into
    lambda_M = 0.6967   /  equal-LENGTH thirds, which is what AHA approximates.

    NOTE: equal-length thirds give segments of UNEQUAL mass -- a known and
    accepted criticism of the AHA standard.  If you later switch to equal-VOLUME
    bands (arguably better for comparing regional work or perfusion per unit
    mass), that is a deliberate DEVIATION from AHA and must be declared.
    Only lambda_A and lambda_M change; nothing in this module needs rewriting.

PARALLEL SAFETY
---------------
Every function here is PURE LOCAL ARITHMETIC on the coordinates handed to it.
There are no MPI reductions, no global searches, no collective operations.

This is deliberate and it is important.  Under MPI, each rank sees only its own
slice of the mesh: no single rank can see both the apex and the base.  If we
tried to compute apex_point or lambda_c at runtime we would get a different
answer on every rank and the segmentation would silently disagree across the
domain.  Instead, the geometry is measured ONCE, OFFLINE, IN SERIAL, and passed
in through the JSON parameter block.  Runtime labelling is then embarrassingly
parallel and cannot go wrong.

USAGE
-----
    params = instruction_data["mesh"]["aha_segmentation"]

    seg_q, lam, theta, n_clamped = segment_gauss(xq, params)      # reporting
    seg_q_elem, seg_cell         = segment_elements(xq, params)   # material
"""
from __future__ import division
import numpy as np


# ===========================================================================
# SECTOR -> AHA SEGMENT ID LOOKUP TABLES
# ===========================================================================
#
# WHY THESE LOOK "BACKWARDS" -- READ BEFORE CHANGING ANYTHING
# -----------------------------------------------------------
# AHA numbers the basal ring 1..6 in this order:
#     1 anterior -> 2 anteroseptal -> 3 inferoseptal -> 4 inferior
#       -> 5 inferolateral -> 6 anterolateral
#
# In the standard bull's-eye display (viewed FROM THE APEX; anterior at the
# top, septum on the LEFT, lateral on the RIGHT, inferior at the bottom) that
# order runs COUNTER-CLOCKWISE.
#
# But our theta runs CLOCKWISE in that same view.  Here is why:
#   - e_l points apex -> base, i.e. AWAY from a viewer standing at the apex.
#   - We build a RIGHT-HANDED frame (e_1, e_2, e_l).
#   - A right-handed rotation about a vector appears COUNTER-clockwise when the
#     vector points TOWARD you, and CLOCKWISE when it points AWAY from you.
#   - Viewed from the apex, e_l points away.  Therefore theta increases
#     clockwise.
#
# Clockwise theta + counter-clockwise AHA numbering  =>  the maps are reversed.
#
# CONSEQUENCE: if you ever flip the sign of axis_vector, theta reverses and
# EVERY map below becomes wrong -- you would get a MIRRORED bull's-eye (e.g.
# segments 2 and 6 swapped).  On an axisymmetric mesh a mirrored bull's-eye
# looks completely plausible; six wedges, correct sizes, correct colours.  YOU
# CANNOT CATCH THIS BY EYE.  Use the marker test in the driver script.
#
# Index into these arrays with the sector number k, NOT with the segment id.

# Basal ring: 6 sectors of 60 degrees.  k = floor(theta_deg / 60)
BASAL_MAP = np.array([1, 6, 5, 4, 3, 2])
#                     ^  ^  ^  ^  ^  ^
#     k=0 -> seg 1  (basal anterior)        theta in [  0,  60)
#     k=1 -> seg 6  (basal anterolateral)   theta in [ 60, 120)
#     k=2 -> seg 5  (basal inferolateral)   theta in [120, 180)
#     k=3 -> seg 4  (basal inferior)        theta in [180, 240)
#     k=4 -> seg 3  (basal inferoseptal)    theta in [240, 300)
#     k=5 -> seg 2  (basal anteroseptal)    theta in [300, 360)

# Mid ring: same 6 sectors, ids shifted by +6.
MID_MAP = np.array([7, 12, 11, 10, 9, 8])
#     k=0 -> seg 7  (mid anterior)
#     k=1 -> seg 12 (mid anterolateral)
#     k=2 -> seg 11 (mid inferolateral)
#     k=3 -> seg 10 (mid inferior)
#     k=4 -> seg 9  (mid inferoseptal)
#     k=5 -> seg 8  (mid anteroseptal)

# Apical ring: only 4 sectors of 90 degrees (the ring is smaller, so AHA uses
# fewer, coarser segments there).
APICAL_MAP = np.array([13, 16, 15, 14])
#     k=0 -> seg 13 (apical anterior)
#     k=1 -> seg 16 (apical lateral)
#     k=2 -> seg 15 (apical inferior)
#     k=3 -> seg 14 (apical septal)

# The apex cap: one segment, no angular subdivision.
APEX_CAP_ID = 17


# ===========================================================================
# THE APICAL ANGULAR OFFSET -- WHERE 15 DEGREES COMES FROM
# ===========================================================================
#
# This is DERIVED, not tuned.  Do not treat it as a free parameter.
#
# We declare theta = 0 at the anterior RV insertion point, which in AHA is the
# BOUNDARY between the anteroseptal and anterior segments.  Therefore:
#
#   - Basal/mid segment 1 (anterior) spans theta in [0, 60), so the ANTERIOR
#     direction itself sits at its centre:  theta = 30 degrees.
#
#   - Apical segment 13 (apical anterior) must be CENTRED on that SAME anterior
#     direction.  It is a 90-degree sector, so it must span
#         [30 - 45, 30 + 45)  =  [-15, +75)
#
#   - To make floor((theta + offset) / 90) return k=0 over [-15, 75), we need
#         offset = +15 degrees.
#
# THE DEEPER POINT: there is only ONE genuine unknown in this whole scheme --
# the anatomical anchor c_ref.  Both ring offsets follow from it by the AHA
# definition.  If you ever expose TWO independently tunable offsets, you can
# fit anything, and you will eventually produce an incoherent segmentation
# (e.g. apical-septal not aligned with the midpoint of the two mid-septal
# segments).  Derive; do not tune.
#
# SANITY CHECK you can see in ParaView: apical segment 14 (septal) should be
# centred exactly on the boundary between basal segments 2 and 3.
APICAL_OFFSET_DEG = 15.0


# ===========================================================================
# CORONARY PERFUSION TERRITORIES
# ===========================================================================
#
# Standard AHA assignment of the 17 segments to the three main coronary
# arteries.  This is the eventual consumer of the segmentation for the
# perfusion constitutive law.
#
# Worth knowing: this 3-way merge is COARSE.  Most of the fussy choices above
# (the exact lambda_c, the 15-degree offset) barely survive it.  Build AHA-17
# anyway because it is the standard and it is reviewable -- but do not agonise
# over sub-degree precision if the physics only sees three territories.
LAD = [1, 2, 7, 8, 13, 14, 17]   # left anterior descending
RCA = [3, 4, 9, 10, 15]          # right coronary artery
LCX = [5, 6, 11, 12, 16]         # left circumflex


# ===========================================================================
# COORDINATE FRAME
# ===========================================================================

def build_frame(params):
    """Build the orthonormal anatomical frame from the JSON parameter block.

    Returns
    -------
    apex : (3,) the point on the long axis at the apical end
    e_l  : (3,) UNIT long-axis vector, pointing APEX -> BASE
    e_1  : (3,) UNIT circumferential reference direction (theta = 0)
    e_2  : (3,) UNIT vector completing the right-handed frame
    L    : float, the apex-to-base distance used to normalise lambda

    The frame (e_1, e_2, e_l) is RIGHT-HANDED by construction, since
    e_2 = e_l x e_1.  This is what fixes the sign of theta -- see the long
    comment above the lookup tables.
    """
    # ---- the apical end of the axis --------------------------------------
    # This is a POINT ON THE AXIS at the apex, measured offline.  Do not try
    # to find "the most apical node" at runtime: on a discretised mesh that is
    # noisy, and under MPI each rank would find a different one.
    apex = np.asarray(params["apex_point"], dtype=float)

    # ---- the long axis, as a VECTOR --------------------------------------
    # Carrying the axis as a vector (not just "use z") is what makes the theta
    # handedness fall out automatically instead of being something you have to
    # remember.  It also means a non-axis-aligned patient-specific mesh needs
    # no code change -- only a different JSON value.
    e_l = np.asarray(params["axis_vector"], dtype=float)
    e_l = e_l / np.linalg.norm(e_l)   # normalise defensively

    # ---- apex-to-base distance -------------------------------------------
    L = float(params["axis_length"])

    # ---- the circumferential reference direction (theta = 0) -------------
    # c_ref as supplied by the user need NOT be perpendicular to the axis.
    # We project out any component along e_l (Gram-Schmidt), because an angle
    # about the axis is only defined in the plane perpendicular to it.
    c_ref = np.asarray(params["c_ref"], dtype=float)
    e_1 = c_ref - np.dot(c_ref, e_l) * e_l
    n1 = np.linalg.norm(e_1)
    if n1 < 1e-12:
        # c_ref was (anti)parallel to the axis -- it has no component in the
        # short-axis plane, so it cannot define an angular origin.
        raise ValueError(
            "c_ref is parallel to axis_vector; it defines no circumferential "
            "reference direction.  Choose a c_ref with a component "
            "perpendicular to the long axis.")
    e_1 = e_1 / n1

    # ---- complete the right-handed frame ---------------------------------
    # ORDER MATTERS.  e_l x e_1, not e_1 x e_l.  Swapping these reverses theta
    # and mirrors the entire bull's-eye.
    e_2 = np.cross(e_l, e_1)

    return apex, e_l, e_1, e_2, L


def lambda_theta(points, params):
    """Map points to anatomical coordinates (lambda, theta).

    Parameters
    ----------
    points : (N, 3) array of positions -- quadrature points or cell centroids
    params : the aha_segmentation JSON block

    Returns
    -------
    lam       : (N,) longitudinal coordinate, clamped to [0, 1]
    theta     : (N,) circumferential angle in [0, 2*pi)
    n_clamped : int, how many points fell outside [0, 1] before clamping

    ON n_clamped
    ------------
    A SMALL count is expected and harmless: floating-point points sitting
    exactly on the base plane, or a hair beyond the epicardial apex.

    A LARGE count is a BUG SIGNAL -- it means apex_point or axis_length is
    wrong, and your whole lambda scale is off, which silently moves every band
    boundary.  Always print this and look at it.  Never ignore it.
    """
    apex, e_l, e_1, e_2, L = build_frame(params)

    # Position of each point RELATIVE TO THE APEX, in the anatomical frame.
    p = np.asarray(points, dtype=float) - apex

    # ---- lambda: normalised projection onto the long axis -----------------
    # p . e_l is the signed distance along the axis from the apex.
    # Dividing by L maps [apex, base] -> [0, 1].
    lam_raw = np.dot(p, e_l) / L

    n_clamped = int(np.sum((lam_raw < 0.0) | (lam_raw > 1.0)))
    lam = np.clip(lam_raw, 0.0, 1.0)

    # ---- theta: angle in the short-axis plane -----------------------------
    # Project p onto the two in-plane basis vectors and take the arctangent.
    # arctan2 (NOT arctan) is essential: it uses the signs of BOTH components
    # to resolve the full circle, giving a result in (-pi, +pi].  Plain arctan
    # would collapse opposite sides of the LV onto the same angle.
    theta = np.arctan2(np.dot(p, e_2), np.dot(p, e_1))

    # Shift from (-pi, pi] to [0, 2*pi) so that the sector binning below can
    # use a simple floor division without worrying about negative angles.
    theta = np.mod(theta, 2.0 * np.pi)

    return lam, theta, n_clamped


# ===========================================================================
# CLASSIFICATION
# ===========================================================================

def classify(lam, theta, params):
    """Map anatomical coordinates (lambda, theta) to AHA segment ids 1..17."""
    lam_c = float(params["lambda_c"])   # apex cap boundary (endocardial apex)
    lam_A = float(params["lambda_A"])   # apical / mid boundary
    lam_M = float(params["lambda_M"])   # mid / basal boundary

    theta_deg = np.degrees(theta)

    # Start at 0, which is NOT a valid AHA id.  Any point still holding 0 at
    # the end was never assigned -- see the guard below.
    seg = np.zeros(lam.shape, dtype=int)

    # ---- longitudinal bands ----------------------------------------------
    # The comparisons are chosen so the four bands are mutually exclusive AND
    # exhaustive over [0, 1]: [0, lam_c) [lam_c, lam_A) [lam_A, lam_M) [lam_M, 1]
    # No point can land in two bands, and none can fall through the cracks.
    apex_cap = lam < lam_c
    apical   = (lam >= lam_c) & (lam < lam_A)
    mid      = (lam >= lam_A) & (lam < lam_M)
    basal    = lam >= lam_M

    # ---- apex cap: no angle used -----------------------------------------
    seg[apex_cap] = APEX_CAP_ID

    # ---- 6-sector rings (basal and mid) ----------------------------------
    # floor(theta/60) gives 0..5 for theta in [0, 360).  The "% 6" is belt-and-
    # braces against a point landing exactly at theta = 360 due to rounding,
    # which would otherwise index out of bounds.
    k6 = np.floor(theta_deg / 60.0).astype(int) % 6
    seg[mid]   = MID_MAP[k6[mid]]
    seg[basal] = BASAL_MAP[k6[basal]]

    # ---- 4-sector ring (apical) ------------------------------------------
    # The +15 offset shifts the 90-degree grid so its sectors are centred on
    # anterior / lateral / inferior / septal -- the same anatomical directions
    # the 60-degree sectors are centred on.  See the long derivation above.
    k4 = np.floor((theta_deg + APICAL_OFFSET_DEG) / 90.0).astype(int) % 4
    seg[apical] = APICAL_MAP[k4[apical]]

    # ---- fail loudly, never silently -------------------------------------
    # A silent zero here would sail straight through into the material
    # assignment and give a region with garbage properties.  Raise instead.
    if np.any(seg == 0):
        raise RuntimeError(
            "%d points were never assigned a segment.  The longitudinal band "
            "definitions do not cover [0, 1] -- check lambda_c, lambda_A, "
            "lambda_M." % int(np.sum(seg == 0)))

    return seg


# ===========================================================================
# PUBLIC ENTRY POINTS
# ===========================================================================

def segment_gauss(xq, params):
    """One AHA id per QUADRATURE POINT.  Use for reporting and visualisation.

    This is the fine-grained labelling.  It resolves segment boundaries down
    to the quadrature level, which is what you want for:
      - dumping CSV and looking at the bull's-eye in ParaView
      - computing volume per segment
      - regional post-processing (strain, work per segment)

    Do NOT use this to assign material properties -- see segment_elements.

    Returns (seg, lam, theta, n_clamped).  The lam and theta arrays are
    returned deliberately: ALWAYS validate those two fields before you trust
    the segment ids.  If lambda or theta is wrong, the ids are wrong, and the
    ids alone will not tell you which.
    """
    lam, theta, n_clamped = lambda_theta(xq, params)
    return classify(lam, theta, params), lam, theta, n_clamped


def segment_elements(xq, params, n_quad_per_cell=4):
    """One AHA id per ELEMENT, from the cell centroid.  Use for MATERIAL.

    WHY NOT JUST USE THE GAUSS-POINT LABELS FOR MATERIAL?
    -----------------------------------------------------
    Because a hard partition at the quadrature level puts a CONSTITUTIVE
    DISCONTINUITY *inside* individual elements: one tet could have 2 Gauss
    points that are "healthy" and 2 that are "infarcted".  That produces
    non-physical stress jumps within the element and can wreck Newton
    convergence.

    Note this is genuinely different from the existing chronic_infarct law,
    which uses a SMOOTH Gaussian falloff -- smooth fields do not have this
    problem.  A bull's-eye is a HARD partition and does.

    So: compute the label ONCE per cell from its centroid, then give all 4 of
    that cell's quadrature points the SAME label.  Every element then has
    exactly one parameter set.  No intra-element discontinuity.

    ORDERING ASSUMPTION
    -------------------
    Assumes the quadrature dofs are CELL-MAJOR: cell c owns quadrature indices
    4c, 4c+1, 4c+2, 4c+3.  Verified on this mesh (n_quad = 5008 = 4 x 1252).
    Quadrature-element dofs are not shared between cells, so this holds -- but
    the reshape below will produce nonsense if it ever stops holding, so the
    driver script cross-checks it.

    The 4-point tetrahedral rule has EQUAL weights, so the mean of the 4
    quadrature coordinates IS the cell centroid, exactly.

    Returns (seg_per_quadrature_point, seg_per_cell).
    """
    xq = np.asarray(xq, dtype=float)
    n = xq.shape[0]

    if n % n_quad_per_cell != 0:
        raise ValueError(
            "n_quad (%d) is not divisible by %d -- the quadrature dof layout "
            "is not what this function assumes." % (n, n_quad_per_cell))

    n_cells = n // n_quad_per_cell

    # (n_quad, 3) -> (n_cells, 4, 3) -> mean over the 4 -> (n_cells, 3)
    cent = xq.reshape(n_cells, n_quad_per_cell, 3).mean(axis=1)

    lam_c, th_c, _ = lambda_theta(cent, params)
    seg_cell = classify(lam_c, th_c, params)

    # Broadcast each cell's label back out to its 4 quadrature points, so the
    # result can be written directly into a quadrature-space dolfin Function.
    seg_q = np.repeat(seg_cell, n_quad_per_cell)

    return seg_q, seg_cell


def territory(seg):
    """Collapse AHA segment ids to coronary territories.

    Returns 1 = LAD, 2 = RCA, 3 = LCX.

    This is what the perfusion constitutive law will actually consume.  Uses
    np.in1d rather than np.isin because the numpy inside the FEniCS 2017
    Singularity image predates np.isin (added in numpy 1.13).
    """
    seg = np.asarray(seg)
    t = np.zeros(seg.shape, dtype=int)
    t[np.in1d(seg, LAD).reshape(seg.shape)] = 1
    t[np.in1d(seg, RCA).reshape(seg.shape)] = 2
    t[np.in1d(seg, LCX).reshape(seg.shape)] = 3
    return t