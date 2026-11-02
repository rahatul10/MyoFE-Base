# -*- coding: utf-8 -*-
"""
Throwaway: label every cell of the LV mesh with its AHA segment, perfusion
region and main vessel, and write the result for ParaView.  Runs locally --
no FEniCS needed, only numpy and h5py.

Labels are assigned per cell (centroid), exactly as segment_elements() does,
so these match what the simulation uses.

Usage (from the dependencies folder, so aha_segmentation imports):
    python3 export_aha_regions.py path/to/ellipsoidal.hdf5

Outputs:
    aha_regions.vtk   the tetrahedral mesh with cell data seg, region, vessel
                      -> open this in ParaView for a solid, colored wall
    aha_regions.csv   one row per cell centroid: x,y,z,seg,region,vessel

    seg     AHA segment 1..17
    region  perfusion region, see the legend printed below
    vessel  1=LAD, 2=RCA, 3=LCX
"""
from __future__ import division, print_function

import sys
import numpy as np
import h5py

import aha_segmentation as aha

MESH = sys.argv[1] if len(sys.argv) > 1 else "ellipsoidal.hdf5"
MESH_NAME = "ellipsoidal"
OUT_VTK = "aha_regions.vtk"
OUT_CSV = "aha_regions.csv"

with h5py.File(MESH, "r") as f:
    if MESH_NAME not in f:
        sys.exit("No '%s' group in %s. Top-level keys: %s"
                 % (MESH_NAME, MESH, list(f.keys())))
    grp = f[MESH_NAME]
    coords = grp["coordinates"][:]
    cells = grp["topology"][:].astype(int)

if cells.shape[1] != 4:
    sys.exit("Expected tetrahedra (4 nodes per cell), got %d." % cells.shape[1])

cent = coords[cells].mean(axis=1)

lam, theta, n_clamped = aha.lambda_theta(cent, aha.AHA_GEOMETRY)
seg = aha.classify(lam, theta, aha.AHA_GEOMETRY)
reg = aha.region_of(seg)
# Same as aha.territory() (1=LAD, 2=RCA, 3=LCX), written here with np.isin so
# it also works on newer numpy, where territory()'s np.in1d has been removed.
ves = np.zeros(seg.shape, dtype=int)
for vid, segs in ((1, aha.LAD), (2, aha.RCA), (3, aha.LCX)):
    ves[np.isin(seg, segs)] = vid

print("nodes   :", len(coords))
print("cells   :", len(seg), "(expect 1252)")
print("clamped :", n_clamped, "(expect 0)")
print("unmapped region:", int(np.sum(reg == 0)),
      " unmapped vessel:", int(np.sum(ves == 0)))
print()
print("region legend:")
legend = aha.region_legend()
for rid in sorted(legend):
    name = legend[rid]
    print("  %d  %-6s  AHA %s  (%d cells)"
          % (rid, name, sorted(aha.PERFUSION_REGIONS[name]), int(np.sum(reg == rid))))
print("vessel legend: 1=LAD, 2=RCA, 3=LCX")

# Legacy VTK unstructured grid: the real tets, one label per cell.
n_nodes = len(coords)
n_cells = len(cells)
with open(OUT_VTK, "w") as out:
    out.write("# vtk DataFile Version 3.0\n")
    out.write("AHA segments and perfusion regions\n")
    out.write("ASCII\n")
    out.write("DATASET UNSTRUCTURED_GRID\n")
    out.write("POINTS %d double\n" % n_nodes)
    for p in coords:
        out.write("%.8f %.8f %.8f\n" % (p[0], p[1], p[2]))
    out.write("CELLS %d %d\n" % (n_cells, 5 * n_cells))
    for c in cells:
        out.write("4 %d %d %d %d\n" % (c[0], c[1], c[2], c[3]))
    out.write("CELL_TYPES %d\n" % n_cells)
    out.write("10\n" * n_cells)                      # 10 = VTK_TETRA
    # FIELD block rather than repeated SCALARS: VTK readers load every FIELD
    # array, but by default only the first SCALARS array.
    out.write("CELL_DATA %d\n" % n_cells)
    out.write("FIELD labels 3\n")
    for name, arr in (("seg", seg), ("region", reg), ("vessel", ves)):
        out.write("%s 1 %d int\n" % (name, n_cells))
        out.write("\n".join("%d" % v for v in arr))
        out.write("\n")

np.savetxt(OUT_CSV, np.column_stack([cent, seg, reg, ves]),
           delimiter=",", comments="", fmt=["%.6f"] * 3 + ["%d"] * 3,
           header="x,y,z,seg,region,vessel")

print()
print("wrote", OUT_VTK, "and", OUT_CSV)
