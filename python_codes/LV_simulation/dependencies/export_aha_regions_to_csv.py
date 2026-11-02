# -*- coding: utf-8 -*-
"""
Throwaway: dump AHA segment / perfusion region / main vessel per cell to CSV
for ParaView.  Runs locally -- no FEniCS needed, only numpy and h5py.

Labels are assigned per cell (centroid), exactly as segment_elements() does,
so these match what the simulation uses.

Usage (from the dependencies folder, so aha_segmentation imports):
    python export_aha_regions_to_csv.py path/to/ellipsoidal.hdf5

Output: aha_regions.csv  with columns x,y,z,seg,region,vessel
    seg     AHA segment 1..17
    region  perfusion region, see region_legend() printed below
    vessel  1=LAD, 2=RCA, 3=LCX
"""
from __future__ import division, print_function

import sys
import numpy as np
import h5py

import aha_segmentation as aha

MESH = sys.argv[1] if len(sys.argv) > 1 else "ellipsoidal.hdf5"
MESH_NAME = "ellipsoidal"
OUT = "aha_regions.csv"

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
ves = aha.territory(seg)

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

np.savetxt(OUT, np.column_stack([cent, seg, reg, ves]),
           delimiter=",", comments="", fmt=["%.6f"] * 3 + ["%d"] * 3,
           header="x,y,z,seg,region,vessel")
print()
print("wrote", OUT)
