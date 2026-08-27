"""
Step 1 of the 0D epicardial coronary network.

Segment table (Wang et al., Table 1) plus a topology check.  No physics here
yet: this file only declares the tree and proves it is a well-formed tree.
Nothing downstream is trustworthy if this part is wrong, so it validates
itself on import and again, more loudly, under __main__.

Units follow the paper: R [kPa s cm^-3], L [kPa s^2 cm^-3], C [cm^3 kPa^-1],
Z [kPa s cm^-3].

TOPOLOGY IS INFERRED FROM Fig. 1(a) AND MUST BE CONFIRMED.  The R/L/C/Z values
below are verified against the Table 1 page image.  The parent/child wiring is
read off the schematic, which is small; check it against the figure before
trusting any result.  See VERIFY notes at the bottom.
"""

from collections import OrderedDict


# ----------------------------------------------------------------------------
# Node naming
# ----------------------------------------------------------------------------
# Nodes are junctions.  "AO" is the aortic root: the single inlet where the
# tree hangs off the systemic circulation (P_AO is imposed there, Eq. 23).
# Terminal nodes are named T_<segment> and carry the microcirculatory
# impedance Z; the averaged intramyocardial pressure is imposed there
# (Eq. 24).  Interior nodes are named N_<something> for readability only --
# the solver never cares about the string, only the connectivity.

INLET_NODE = "AO"


# ----------------------------------------------------------------------------
# Segment table -- Table 1, verified against the rendered page.
# ----------------------------------------------------------------------------
# Each entry: (anatomical name, R, L, C, Z, proximal node, distal node)
# Z is None for segments that branch further (no terminal impedance).

SEGMENTS = OrderedDict([
    # --- left side -----------------------------------------------------
    ("LMCA",  dict(name="Left main coronary artery",
                   R=0.2156, L=0.0227, C=0.0003,   Z=None,
                   node_p="AO",       node_d="N_LMCA")),

    ("LAD",   dict(name="Left anterior descending artery",
                   R=0.4299, L=0.0294, C=0.0002,   Z=None,
                   node_p="N_LMCA",   node_d="N_LAD")),

    ("LAD1",  dict(name="LAD segment 1",
                   R=0.4944, L=0.0327, C=0.0002,   Z=145.6,
                   node_p="N_LAD",    node_d="T_LAD1")),

    ("LAD2",  dict(name="LAD segment 2",
                   R=2.3796, L=0.1107, C=0.0003,   Z=None,
                   node_p="N_LAD",    node_d="N_LAD2")),

    ("LAD3",  dict(name="LAD segment 3",
                   R=3.1904, L=0.1058, C=0.0001,   Z=155.0,
                   node_p="N_LAD2",   node_d="T_LAD3")),

    ("LAD4",  dict(name="LAD segment 4",
                   R=3.6283, L=0.0772, C=0.00001,  Z=80.2,
                   node_p="N_LAD2",   node_d="T_LAD4")),

    ("LCX",   dict(name="Left circumflex artery",
                   R=0.3390, L=0.0230, C=0.0001,   Z=None,
                   node_p="N_LMCA",   node_d="N_LCX")),

    ("MARG1", dict(name="Marginal branch 1",
                   R=1.6994, L=0.0666, C=0.0001,   Z=155.0,
                   node_p="N_LCX",    node_d="T_MARG1")),

    ("LCX1",  dict(name="LCX segment 1",
                   R=0.4215, L=0.0224, C=0.0001,   Z=None,
                   node_p="N_LCX",    node_d="N_LCX1")),

    ("MARG2", dict(name="Marginal branch 2",
                   R=2.5890, L=0.0761, C=0.0001,   Z=168.4,
                   node_p="N_LCX1",   node_d="T_MARG2")),

    ("LCX2",  dict(name="LCX segment 2",
                   R=0.8890, L=0.0364, C=0.0001,   Z=None,
                   node_p="N_LCX1",   node_d="N_LCX2")),

    ("MARG3", dict(name="Marginal branch 3",
                   R=2.7165, L=0.0866, C=0.0001,   Z=72.0,
                   node_p="N_LCX2",   node_d="T_MARG3")),

    ("LCX3",  dict(name="LCX segment 3",
                   R=2.9208, L=0.0922, C=0.0001,   Z=170.1,
                   node_p="N_LCX2",   node_d="T_LCX3")),

    # --- right side ----------------------------------------------------
    ("RCA",   dict(name="Right coronary artery",
                   R=1.6671, L=0.1164, C=0.0006,   Z=None,
                   node_p="AO",       node_d="N_RCA")),

    ("PLA",   dict(name="Posterolateral artery",
                   R=1.2531, L=0.0627, C=0.0002,   Z=59.81,
                   node_p="N_RCA",    node_d="T_PLA")),

    ("PDA",   dict(name="Posterior descending artery",
                   R=2.1974, L=0.0779, C=0.0001,   Z=139.72,
                   node_p="N_RCA",    node_d="T_PDA")),
])


# ----------------------------------------------------------------------------
# Derived sets
# ----------------------------------------------------------------------------

def outlet_segments():
    """Segments that terminate in myocardium (have a Z).  These are the ones
    whose distal pressure is set to the territory-averaged IMP (Eq. 24)."""
    return [s for s, d in SEGMENTS.items() if d["Z"] is not None]


def inlet_segments():
    """Segments whose proximal node is the aortic root (Eq. 23)."""
    return [s for s, d in SEGMENTS.items() if d["node_p"] == INLET_NODE]


def all_nodes():
    nodes = set()
    for d in SEGMENTS.values():
        nodes.add(d["node_p"])
        nodes.add(d["node_d"])
    return nodes


def interior_nodes():
    """Nodes where Kirchhoff's laws apply (Eq. 5): not the inlet, not terminal."""
    return sorted(n for n in all_nodes()
                  if n != INLET_NODE and not n.startswith("T_"))


# ----------------------------------------------------------------------------
# Topology validation
# ----------------------------------------------------------------------------

def check_topology(verbose=False):
    """Prove the segment table describes a single connected tree rooted at AO.

    Raises AssertionError on any structural defect.  Returns a small report
    dict so the caller can print counts.
    """
    errors = []

    # -- 1. every non-inlet segment's proximal node must be some other
    #       segment's distal node, i.e. no orphan branches.
    distal_nodes = {d["node_d"] for d in SEGMENTS.values()}
    for s, d in SEGMENTS.items():
        if d["node_p"] != INLET_NODE and d["node_p"] not in distal_nodes:
            errors.append(
                "segment %s hangs off node %s, which no segment feeds"
                % (s, d["node_p"]))

    # -- 2. a tree has exactly one parent per node: no node may be the
    #       distal end of two different segments.
    seen = {}
    for s, d in SEGMENTS.items():
        nd = d["node_d"]
        if nd in seen:
            errors.append("node %s is the distal end of both %s and %s"
                          % (nd, seen[nd], s))
        seen[nd] = s

    # -- 3. no self-loops
    for s, d in SEGMENTS.items():
        if d["node_p"] == d["node_d"]:
            errors.append("segment %s is a self-loop at %s" % (s, d["node_p"]))

    # -- 4. reachability: walk from AO and confirm every segment is reached.
    #       This catches disconnected sub-trees that rules 1-3 would miss.
    reached_nodes = {INLET_NODE}
    reached_segs = set()
    changed = True
    while changed:
        changed = False
        for s, d in SEGMENTS.items():
            if s in reached_segs:
                continue
            if d["node_p"] in reached_nodes:
                reached_segs.add(s)
                reached_nodes.add(d["node_d"])
                changed = True
    unreached = set(SEGMENTS) - reached_segs
    if unreached:
        errors.append("segments unreachable from %s: %s"
                      % (INLET_NODE, sorted(unreached)))

    # -- 5. terminal/interior consistency.  A node with no outgoing segment
    #       is a leaf; every leaf must belong to a segment carrying a Z, and
    #       every segment carrying a Z must end at a leaf.
    has_outgoing = {d["node_p"] for d in SEGMENTS.values()}
    for s, d in SEGMENTS.items():
        is_leaf = d["node_d"] not in has_outgoing
        has_Z = d["Z"] is not None
        if is_leaf and not has_Z:
            errors.append("segment %s ends at leaf %s but has no terminal "
                          "impedance Z" % (s, d["node_d"]))
        if has_Z and not is_leaf:
            errors.append("segment %s has Z but continues into the tree"
                          % s)

    # -- 6. naming convention: leaves should be T_*, so the assembly code can
    #       identify outlet rows by name without re-deriving the topology.
    for s, d in SEGMENTS.items():
        is_leaf = d["node_d"] not in has_outgoing
        if is_leaf and not d["node_d"].startswith("T_"):
            errors.append("leaf node %s (segment %s) should be named T_*"
                          % (d["node_d"], s))

    # -- 7. positivity: R, L, C must be strictly positive or the discrete
    #       system is singular / unphysical.
    for s, d in SEGMENTS.items():
        for k in ("R", "L", "C"):
            if d[k] <= 0.0:
                errors.append("segment %s has non-positive %s = %r"
                              % (s, k, d[k]))
        if d["Z"] is not None and d["Z"] <= 0.0:
            errors.append("segment %s has non-positive Z = %r" % (s, d["Z"]))

    if errors:
        raise AssertionError(
            "coronary topology is not a well-formed tree:\n  - "
            + "\n  - ".join(errors))

    report = dict(
        n_segments=len(SEGMENTS),
        n_outlets=len(outlet_segments()),
        n_inlets=len(inlet_segments()),
        n_nodes=len(all_nodes()),
        n_interior=len(interior_nodes()),
        n_unknowns=4 * len(SEGMENTS),
    )

    if verbose:
        for k in sorted(report):
            print("  %-12s %s" % (k, report[k]))

    return report


# Validate on import.  A malformed table should fail loudly and immediately
# rather than producing plausible-looking flow waveforms later.
check_topology()


# ----------------------------------------------------------------------------
if __name__ == "__main__":

    print("Coronary segment table -- Wang et al. Table 1")
    print("=" * 68)
    print("%-7s %9s %9s %10s %9s   %-9s -> %-9s"
          % ("seg", "R", "L", "C", "Z", "prox", "dist"))
    print("-" * 68)
    for s, d in SEGMENTS.items():
        z = "-" if d["Z"] is None else "%.2f" % d["Z"]
        print("%-7s %9.4f %9.4f %10.5f %9s   %-9s -> %-9s"
              % (s, d["R"], d["L"], d["C"], z, d["node_p"], d["node_d"]))

    print()
    print("Topology check")
    print("=" * 68)
    rep = check_topology(verbose=True)
    print()

    print("Inlet segments (P_p := P_AO,  Eq. 23):")
    print("  " + ", ".join(inlet_segments()))
    print()

    outs = outlet_segments()
    print("Outlet segments (P_d := P_IMP^s,  Eq. 24)   [%d of them]:" % len(outs))
    print("  " + ", ".join(outs))
    print()

    print("Tree structure:")
    children = {}
    for s, d in SEGMENTS.items():
        children.setdefault(d["node_p"], []).append(s)

    def walk(node, depth):
        for s in children.get(node, []):
            d = SEGMENTS[s]
            leaf = "" if d["Z"] is None else "   <-- terminal (Z=%.2f)" % d["Z"]
            print("  " + "    " * depth + "+- " + s + leaf)
            walk(d["node_d"], depth + 1)

    print("  " + INLET_NODE)
    walk(INLET_NODE, 0)
    print()
    print("OK: %d segments, %d terminals, %d unknowns in Z."
          % (rep["n_segments"], rep["n_outlets"], rep["n_unknowns"]))
