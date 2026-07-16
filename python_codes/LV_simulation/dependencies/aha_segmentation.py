# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 2026

@author: Islam, Rahatul

AHA 17-segment bull's-eye segmentation for the LV.

    position -> (lambda, theta) -> longitudinal band -> circumferential sector
             -> AHA segment id (1..17)

The label is a function of position, so it is mesh-independent and survives
refinement.

CONVENTIONS (coupled -- do not change one without the others):
    lambda  0 at the apex, 1 at the base
    e_l     unit long-axis vector, apex -> base
    theta   right-handed about e_l, measured from c_ref, in [0, 2*pi)
    theta = 0 is DECLARED to be the anterior RV insertion point

This mesh is an axisymmetric LV with no RV, so no direction in the geometry is
intrinsically "anterior".  theta = 0 is a convention imposed via c_ref, not
anatomy recovered from the mesh.  Say so explicitly in any paper.

Segment 17 (apex cap) uses lambda only -- theta degenerates near the apex.

GEOMETRY, measured offline in serial from
demos/HCM_paper/simulations_baseline_final/baseline_final_Kappa_4_k1_3.7:
    base plane z = 0, epicardial apex z = -0.7315, endocardial apex z = -0.6655
    apex_point (0,0,-0.7315), axis_vector +z, axis_length 0.7315
    lambda_c = 0.0902  -- the endocardial apex, where the cavity closes.  This
                          is the AHA definition of the apex cap, not a guess.
    lambda_A = 0.3935, lambda_M = 0.6967  -- equal-length thirds of the
                          cavity-bearing region, which is what AHA approximates.
                          These give segments of unequal mass; that is a known
                          property of the standard, not a bug.

Everything here is pure local arithmetic -- no MPI reductions, no global
searches.  Under MPI no single rank can see both the apex and the base, so the
geometry is measured once, offline, and passed in as parameters.
"""

from __future__ import division

import numpy as np


# Sector index k -> AHA segment id.
#
# These are REVERSED, and they must be.  AHA numbers 1..6 run anterior ->
# anteroseptal -> inferoseptal -> inferior -> inferolateral -> anterolateral,
# which is counter-clockwise in the standard apex-view bull's-eye.  But with
# e_l pointing apex->base (away from a viewer at the apex) and a right-handed
# frame, theta increases CLOCKWISE in that same view.
#
# Flip the sign of axis_vector and every map here becomes wrong -- you get a
# mirrored bull's-eye, which on an axisymmetric mesh looks entirely plausible
# and CANNOT be caught by eye.  Verify with a marker test.
BASAL_MAP = np.array([1, 6, 5, 4, 3, 2])        # k = floor(theta_deg / 60)
MID_MAP = np.array([7, 12, 11, 10, 9, 8])
APICAL_MAP = np.array([13, 16, 15, 14])         # k = floor((theta_deg + 15) / 90)

APEX_CAP_ID = 17

# Derived, not tuned.  theta = 0 sits at the anteroseptal/anterior boundary, so
# anterior is at theta = 30 (centre of segment 1's 60-degree sector).  Apical
# segment 13 must be centred on that same direction, so its 90-degree sector
# spans [-15, 75) -- hence the offset.  There is only ONE genuine unknown here,
# the anchor c_ref; both ring offsets follow from it.  Never tune this.
APICAL_OFFSET_DEG = 15.0

LAD = [1, 2, 7, 8, 13, 14, 17]
RCA = [3, 4, 9, 10, 15]
LCX = [5, 6, 11, 12, 16]


# ---------------------------------------------------------------------------
# PERFUSION REGIONS (coronary sub-territories)
# ---------------------------------------------------------------------------
# Fixed mapping of AHA segments to named coronary perfusion regions, following
# the published branch-level territory table (LAD1/3/4, MARG1/2/3, LCX3, PDA).
# This is anatomy, not a tunable parameter -- it lives in code, not JSON.
#
# Every AHA segment 1..17 belongs to exactly one region.  The assertion below
# enforces that: edit the table and break the partition, and import fails loudly.
PERFUSION_REGIONS = {
    "LAD1":  [1, 2, 8],
    "LAD4":  [7],
    "LAD3":  [13, 14, 17],
    "MARG1": [6, 12],
    "MARG2": [5, 11],
    "MARG3": [15, 16],
    "LCX3":  [4, 10],
    "PDA":   [3, 9],
}

# Region name -> integer ID, assigned by SORTED name so the IDs are
# deterministic and reproducible (Python 2.7 dict order is not).  Do not read
# these numbers from memory -- use region_legend() to recover name from ID.
REGION_NAME_TO_ID = {
    name: i + 1 for i, name in enumerate(sorted(PERFUSION_REGIONS))
}

# Flat lookup: AHA segment id -> region id.  Built once at import.
_SEGMENT_TO_REGION_ID = {}
for _name, _segs in PERFUSION_REGIONS.items():
    for _s in _segs:
        _SEGMENT_TO_REGION_ID[_s] = REGION_NAME_TO_ID[_name]

# Build-time partition check: exactly segments 1..17, none missing or doubled.
_assigned = sorted(_SEGMENT_TO_REGION_ID.keys())
if _assigned != list(range(1, 18)):
    raise RuntimeError(
        "PERFUSION_REGIONS is not a clean partition of AHA segments 1..17. "
        "Got %s." % _assigned)


def build_frame(params):
    """Return the apex, anatomical basis vectors, and axis length."""

    apex = np.asarray(params["apex_point"], dtype=float)

    e_l = np.asarray(params["axis_vector"], dtype=float)
    axis_norm = np.linalg.norm(e_l)

    if axis_norm < 1.0e-12:
        raise ValueError("axis_vector must be nonzero.")

    e_l = e_l / axis_norm

    length = float(params["axis_length"])

    if length <= 0.0:
        raise ValueError("axis_length must be positive.")

    # c_ref need not be perpendicular to the axis; project out any axial part.
    c_ref = np.asarray(params["c_ref"], dtype=float)

    e_1 = c_ref - np.dot(c_ref, e_l) * e_l
    ref_norm = np.linalg.norm(e_1)

    if ref_norm < 1.0e-12:
        raise ValueError(
            "c_ref is parallel to axis_vector. Choose a circumferential "
            "reference direction perpendicular to the axis."
        )

    e_1 = e_1 / ref_norm

    # Order matters: e_l x e_1, not e_1 x e_l.  Swapping these mirrors theta.
    e_2 = np.cross(e_l, e_1)

    return apex, e_l, e_1, e_2, length


def lambda_theta(points, params):
    """Longitudinal position lambda and circumferential angle theta.

    A large n_clamped means apex_point or axis_length is wrong, which silently
    shifts every band boundary.  Always check it.
    """

    apex, e_l, e_1, e_2, length = build_frame(params)

    rel = np.asarray(points, dtype=float) - apex

    lambda_raw = np.dot(rel, e_l) / length
    n_clamped = int(np.sum((lambda_raw < 0.0) | (lambda_raw > 1.0)))
    lam = np.clip(lambda_raw, 0.0, 1.0)

    # arctan2, not arctan -- it uses both signs to resolve the full circle.
    theta = np.arctan2(np.dot(rel, e_2), np.dot(rel, e_1))
    theta = np.mod(theta, 2.0 * np.pi)

    return lam, theta, n_clamped


def classify(lam, theta, params):
    """Assign AHA segment IDs from 1 to 17."""

    lam = np.asarray(lam, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if lam.shape != theta.shape:
        raise ValueError("lam and theta must have the same shape.")

    lambda_c = float(params["lambda_c"])
    lambda_a = float(params["lambda_A"])
    lambda_m = float(params["lambda_M"])

    if not 0.0 <= lambda_c <= lambda_a <= lambda_m <= 1.0:
        raise ValueError("Expected 0 <= lambda_c <= lambda_A <= lambda_M <= 1.")

    theta_deg = np.degrees(theta)
    segment = np.zeros(lam.shape, dtype=int)

    # Bands are mutually exclusive and exhaustive over [0, 1].
    apex_cap = lam < lambda_c
    apical = (lam >= lambda_c) & (lam < lambda_a)
    middle = (lam >= lambda_a) & (lam < lambda_m)
    basal = lam >= lambda_m

    segment[apex_cap] = APEX_CAP_ID

    sector_6 = np.floor(theta_deg / 60.0).astype(int) % 6
    segment[middle] = MID_MAP[sector_6[middle]]
    segment[basal] = BASAL_MAP[sector_6[basal]]

    sector_4 = np.floor((theta_deg + APICAL_OFFSET_DEG) / 90.0).astype(int) % 4
    segment[apical] = APICAL_MAP[sector_4[apical]]

    # A silent zero would sail straight into the material assignment.
    if np.any(segment == 0):
        raise RuntimeError(
            "%d points were not assigned to an AHA segment."
            % int(np.sum(segment == 0))
        )

    return segment


def segment_gauss(xq, params):
    """One AHA segment ID per quadrature point.  For reporting and ParaView.

    Validate lam and theta before trusting the segment IDs -- if the coordinates
    are wrong the IDs are wrong, and the IDs alone will not tell you which.
    """

    lam, theta, n_clamped = lambda_theta(xq, params)
    segment = classify(lam, theta, params)

    return segment, lam, theta, n_clamped


def segment_elements(xq, params, n_quad_per_cell=4):
    """One centroid-based AHA segment ID per element.  For material assignment.

    Labelling at the Gauss level would put a constitutive discontinuity *inside*
    elements (2 points healthy, 2 infarcted in one tet), giving spurious stress
    jumps and possible Newton trouble.  A hard partition has this problem; the
    existing smooth Gaussian infarct law does not.

    Assumes cell-major quadrature dofs (cell c owns 4c..4c+3), verified on this
    mesh: 5008 = 4 x 1252.  The 4-point tet rule has equal weights, so the mean
    of the 4 quadrature coordinates is exactly the centroid.
    """

    xq = np.asarray(xq, dtype=float)

    if xq.ndim != 2 or xq.shape[1] != 3:
        raise ValueError("xq must have shape (number_of_points, 3).")

    if n_quad_per_cell <= 0:
        raise ValueError("n_quad_per_cell must be positive.")

    n_points = xq.shape[0]

    if n_points % n_quad_per_cell != 0:
        raise ValueError(
            "The number of quadrature points (%d) is not divisible by %d."
            % (n_points, n_quad_per_cell)
        )

    n_cells = n_points // n_quad_per_cell

    centroids = xq.reshape(n_cells, n_quad_per_cell, 3).mean(axis=1)

    lam, theta, _ = lambda_theta(centroids, params)
    segment_per_cell = classify(lam, theta, params)
    segment_per_point = np.repeat(segment_per_cell, n_quad_per_cell)

    return segment_per_point, segment_per_cell


def territory(segment):
    """Map AHA segments to 1=LAD, 2=RCA, 3=LCX.  For the perfusion law.

    np.in1d rather than np.isin -- the numpy in the FEniCS 2017 image predates it.
    """

    segment = np.asarray(segment)
    result = np.zeros(segment.shape, dtype=int)

    result[np.in1d(segment, LAD).reshape(segment.shape)] = 1
    result[np.in1d(segment, RCA).reshape(segment.shape)] = 2
    result[np.in1d(segment, LCX).reshape(segment.shape)] = 3

    return result


def region_of(segment):
    """Map AHA segment ids to perfusion-region ids (1..N, by sorted name).

    segment : array of AHA ids in 1..17.  Returns int array of the same shape.
    Use region_legend() to recover the region name from an id.
    """
    segment = np.asarray(segment)
    result = np.zeros(segment.shape, dtype=int)
    for s, rid in _SEGMENT_TO_REGION_ID.items():
        result[segment == s] = rid
    return result


def region_legend():
    """Return {region_id: region_name} for interpreting the region field.

    Print this alongside the region field so the integer labels in ParaView can
    be read back as LAD1, MARG2, PDA, etc.
    """
    return {rid: name for name, rid in REGION_NAME_TO_ID.items()}