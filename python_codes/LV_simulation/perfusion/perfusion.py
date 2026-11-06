"""
Perfusion module: drives the 0D coronary tree from the running simulation.

    P_AO  <- systemic circulation, circ.data['pressure_aorta']   (Eq. 23)
    P_IMP <- territory-averaged intramyocardial pressure         (Eq. 24)

Both enter the coronary tree as boundary conditions on the 0D circuit.  The
mechanics never see the coronary model, so nothing here can destabilise a
solve.  Flow feedback onto contractility (eta, Eqs. 19-20) comes later.

WHERE P_IMP COMES FROM
----------------------
Three sources, chosen in the JSON with "imp_source":

  "multiplier"  the incompressibility Lagrange multiplier p, volume-averaged
                over each coronary territory.  ("mesh" is accepted as an
                alias for backward compatibility.)
  "stress"      one third the trace of the Cauchy stress, i.e. the mean
                normal stress, volume-averaged the same way.  This is Eq. 17
                of Wang et al.
  "radial"      minus the radial component of the Cauchy stress -- the part
                that actually presses on vessels running through the wall.
  "prescribed"  a half-sine over systole with peaks from "imp_peak".  A
                placeholder, kept so the three can be compared directly.

HOW THE TWO MESH SOURCES DIFFER
-------------------------------
MyoFE enforces incompressibility with a Lagrange multiplier rather than a
penalty: forms.py carries -p*(J - 1) in the strain energy, which puts
-p*C^-1 in the PK2 and therefore -p*I in the Cauchy stress.  So

    sigma = sigma_passive + sigma_active - p*I

    "multiplier"  ->  p
    "stress"      ->  -tr(sigma)/3  =  p - tr(sigma_pas + sigma_act)/3

They differ by the trace of the passive and active stress.  Active stress is
uniaxial along the fibre, so its trace is large during systole and the two
sources diverge most when contraction is strongest.

For "stress" the total PK2 comes from mesh.py:

    functions['total_stress'] = Pactive + passive_total_stress

and the Cauchy trace follows from tr(sigma) = tr(F S F^T)/J, which is the
same quantity Eq. 17 integrates since P F^T = F S F^T.

WHY A VOLUME AVERAGE
--------------------
One terminal resistance stands in for many parallel microvascular beds, each
draining a small piece of tissue at its own local pressure:

    Q = sum_i (P_node - p_i)/R_i

Collapsing that to a single resistor Q = (P_node - Pbar)/Rt with
1/Rt = sum_i 1/R_i and solving for Pbar gives the conductance-weighted mean.
A larger piece of tissue holds proportionally more parallel vessels, so
1/R_i scales with volume and the weighted mean becomes

    Pbar = (1/V) * integral of p over the territory

So the volume average is not an approximation of convenience -- it is the
reduction that makes the lumped terminal reproduce the distributed bed.  It
assumes vessel density is uniform within a territory, which is worth stating.

The averaging is done with assemble() over a cell subdomain rather than by
hand over quadrature points, for two reasons: assemble weights each point by
its quadrature weight and Jacobian (an unweighted mean over Gauss points
over-represents small elements), and assemble performs the MPI reduction
internally.  Under MPI no single rank holds a whole territory.

UNITS
-----
Everything the coronary tree touches is mmHg.  The one exception is the
multiplier, which lives in the mechanics and is in Pa; PA_TO_MMHG converts it
at the boundary and nowhere else.

Python 2.7 compatible.
"""

import numpy as np

from dolfin import assemble

from .coronary_rc import (CoronaryRC, SUBTREES, EXTRA_LV_TERMINALS,
                          TERMINAL_RESISTANCE_PHYSIO, narrowing_for_flow,
                          REF_MEAN_P_AO, REF_MEAN_IMP, AUTOREG_P_UPPER,
                          autoregulation_bounds, autoregulation_update,
                          autoregulated_state)

# The AHA segmentation, its geometry, and the segment -> territory map all come
# from dependencies/aha_segmentation.py.  Nothing regional is defined in this
# file, so the territories used for perfusion are exactly the ones validated
# in ParaView (aha_points.csv).
try:
    from ..dependencies import aha_segmentation as aha
except (ImportError, ValueError):
    import aha_segmentation as aha


# ---------------------------------------------------------------------------
# Perfusion -> cross-bridge coupling (detachment version described first;
# the attachment version, "attachment_coupling", scales k_3 instead)
# ---------------------------------------------------------------------------
# Hypothesis: no blood -> no ATP -> no detachment.  Myosin must bind ATP to
# release actin at every strain, and MyoSim lumps that step into its
# detachment rate (kinetics.py, Eq. 16).  Notation:
#
#     p_i      relative perfusion of LV territory i
#     k_f,i    detachment scaling factor of territory i
#     k_det    detachment rate coefficient,  k_det(x) = k_4,0 + k_4,1 x^4
#     J_4,i    cross-bridge detachment flux in territory i
#
# Once per beat, in each LV territory i:
#
#     p_i   = clip( Qbar_i / Q_ref,i , 0, 1 )      beat-mean flow / healthy flow
#     k_f,i -> k_f,i + (c + (1 - c) p_i - k_f,i) / n_tau
#     k_4,0 -> k_f,i * k_4,0      k_4,1 -> k_f,i * k_4,1     at every point in i
#
# so that kinetics.py, unchanged, computes
#
#     J_4,i(x) = min( max_rate, k_f,i * k_det(x) ) * M_FG(x)
#
# p_i = 1 gives back the original MyoSim; p_i = 0 gives k_f,i = c.  Both rates are
# scaled because k_4,0 carries most of the detachment (78% of it in an
# isometric beat): attached heads sit near x = 0, where k_4,1 x^4 is small.
#
# Slower detachment keeps heads attached: force rises and relaxation slows.
# That raises IMP and cuts flow further, so the loop is REINFORCING.  The
# floor c and the n_tau-beat lag keep it stable; below k_f,i ~ 0.4 a half-
# sarcomere no longer relaxes before the next beat.
#
# Everything runs inside this module: LV_simulation.py is not changed.
#
# JSON (inside "perfusion"; every key optional), ONE of:
#     "attachment_coupling": [{"c": 0.0, "n_tau": 3, "t_start_s": 5}]
#         k_f scales k_3 instead: less flow -> fewer heads attach -> less
#         force.  k_f = c + (1 - c) p_i with c = 0 gives p = 1% -> ~0 force.
#     "detachment_coupling": [{"c": 0.5, "n_tau": 3, "t_start_s": 5}]
#         k_f scales k_4,0 and k_4,1 as described above.
# Leave both out and the coupling is off: nothing in MyoSim is touched.
#
# Setting how much perfusion a vessel gets:
#   1. "perfusion_percent": [{"LAD": 20}] -- THE SIMPLE WAY.  LAD carries 20%
#      of its normal flow (100 = healthy).  The code finds the narrowing that
#      gives this in the healthy heart and applies it; p_i is then MEASURED
#      from the coronary tree, so flow still answers to the heart.  With
#      autoregulation on, 20 means 20% AFTER the LAD bed has dilated as far
#      as it can (see AUTOREGULATION below).
#   2. "stenosis": [{"LAD": 80}] -- the same narrowing given directly as
#      percent AREA lost (80 area -> about 89% flow).
#   3. "p_prescribed": {"LAD1": 0.0, "LAD3": 0.5} inside either coupling
#      -- fix p_i directly (0 = no flow, 1 = normal).  Named territories use
#      this value instead of the measured one; the others stay measured.
#      The coronary tree still runs and its flows are still written, but
#      they no longer drive those territories.
#
# Q_ref: beat-mean flow per LV territory in the healthy run B baseline
# (data_8.csv, t = 65.9-79.2 s, 15 beats), mL/min.  Tied to the heart AND to
# TERMINAL_RESISTANCE_PHYSIO: recalibrate the resistances and these must be
# re-measured.  Override per territory with "q_ref": {"LAD1": ...}.
DEFAULT_Q_REF = {
    "LAD1": 27.36, "LAD3": 15.91, "LAD4": 8.48,
    "MARG1": 17.09, "MARG2": 17.91, "MARG3": 13.28,
    "LCX3": 17.91, "PDA": 17.07,
}

COUPLING_KEYS = ('c', 'n_tau', 't_start_s', 'q_ref', 'p_prescribed',
                 'p_reference')

# What p_i is measured against, "p_reference" inside either coupling:
#   "open_artery" (default)  flow now / flow the same territory would get
#       NOW, at this beat's mean aortic pressure and IMPs, with every vessel
#       open.  Only a narrowing lowers p: a territory behind an open artery
#       always has p = 1, however low the pressure goes.  This stands in for
#       autoregulation -- healthy myocardium keeps its flow -- and stops the
#       runaway where a weak heart lowers aortic pressure, which lowers flow
#       everywhere, which weakens the whole heart.  Exact, because the tree is
#       linear: beat-mean flows are its steady solution at beat-mean pressures.
#   "healthy"  flow now / Q_ref, the healthy baseline flow (the original
#       definition).  Any drop in pressure counts as lost perfusion.
# With autoregulation on, "healthy" is the default and "open_artery" is
# refused: autoregulation itself now keeps healthy territories at Q_ref, and
# the open tree has none, so the two references would disagree.
P_REFERENCES = ('open_artery', 'healthy')

# ---------------------------------------------------------------------------
# AUTOREGULATION (the law, its bounds and their sources: coronary_rc.py)
# ---------------------------------------------------------------------------
# ON BY DEFAULT -- it is part of the heart, like the baroreflex, so nothing
# needs to be written.  Each beat every territory checks its own flow: normal
# flow, nothing changes (R/R0 stays 1); low flow, its bed dilates; high flow,
# it constricts.  The only thing a case sets is "perfusion_percent".
# To change a value, inside "perfusion":
#     "autoregulation": [{"cfr": 3.3, "n_ar": 5}]
# To switch it OFF (fixed resistances, as in runs before Nov 2026 -- e.g. to
# show what autoregulation contributes, or to reproduce an old run):
#     "autoregulation": [false]
#
#   cfr        coronary flow reserve, R0/R_min: the bed can dilate to R0/cfr.
#              3.3 = normal (Garcia et al. 2009, Table 1; Rigo 2005).
#   n_ar       beats: each beat the resistance closes 1/n_ar of the gap to the
#              value that would have given Q_ref.  5 (default) gives a
#              half-response of 4.5-5.0 s after a 20-30 mmHg pressure drop
#              (Dankelman et al. 1989: 4.9 s); 6 gives 5.3-5.9 s.
#   p_upper    mmHg up to which flow is held (default 120, Goodwill et al.
#              2017); sets R_max.
#   t_start_s  first beat that adapts (default 0).
#
# Every terminal adapts, PLA included.  Targets: the LV territories' Q_ref
# (the coupling's, so a territory held on target has p = 1 exactly); PLA its
# healthy flow at the reference pressures.  The run STARTS adapted -- the
# resistances are set to where autoregulation settles for the given
# narrowing at the reference pressures -- so there is no start-up transient;
# from then on they follow the heart beat by beat.
#
# Written every step: R_over_R0_<terr> = R_i / R0_i
#     1 = resting tone,  1/cfr (0.303) = fully dilated,  above 1 = constricted.
AUTOREG_KEYS = ('cfr', 'n_ar', 'p_upper', 't_start_s')

# The two couplings: which MyoSim rates k_f multiplies, and the default floor c.
#   "attachment_coupling"  k_3            -> J_3 = r_3 M_DRX (n_on - n_bound)
#                          less flow, fewer heads attach, LESS force; c = 0
#                          by default, so p = 0 means no attachment at all.
#   "detachment_coupling"  k_4,0, k_4,1   -> J_4 = r_4 M_FG
#                          less flow, heads stay attached, MORE force and
#                          slower relaxation; c = 0.5 by default (below
#                          k_f ~ 0.4 the muscle stops relaxing).
# Use one at a time.
COUPLING_TYPES = {
    'attachment_coupling': dict(rates=('k_3',), c=0.0, what='attachment'),
    'detachment_coupling': dict(rates=('k_4_0', 'k_4_1'), c=0.5,
                                what='detachment'),
}

# ---------------------------------------------------------------------------
# Territory outputs: per LV territory, every timestep, into data.csv
# ---------------------------------------------------------------------------
#     J3_<terr>             mean cross-bridge attachment flux
#                               J_3,i = sum_x r_3(x) M_DRX (n_on - n_bound),
#                           fraction of myosin heads attaching per second (1/s),
#                           from kinetics.py's own fluxes['J_3'], so it follows
#                           k_f under the attachment coupling.
#     k_3_<terr>            attachment rate k_3 in use (1/s) = k_f * base
#     J4_<terr>             mean cross-bridge detachment flux
#                               J_4,i = sum_x min(max_rate, k_f,i k_det(x)) M_FG(x)
#                           fraction of myosin heads detaching per second (1/s).
#                           Read from kinetics.py's own fluxes['J_4'] (it calls
#                           the capped, scaled k_det "r_4"), so it follows k_f.
#     k_4_0_<terr>          k_4,0 in use (1/s) = k_f * base
#     k_4_1_<terr>          x^4 detachment coefficient in use       = k_f * base
#     k_1_<terr>            SRX -> DRX rate in use (the baroreflex moves it)
#     active_stress_<terr>  cross-bridge (active) stress along the fibre, Pa --
#                           LV_simulation's cb_stress_list, so one timestep old
# Each is the volume-weighted mean over the territory's integration points
# (reference cell volumes), summed over all MPI ranks.  OFF unless asked for,
# inside "perfusion" -- pick the ones you want:
#     "territory_outputs": [{"J3": true, "J4": true, "k_3": true,
#                            "k_4_0": true, "k_4_1": true, "k_1": true,
#                            "active_stress": true}]
# Fields left out (or false) are not saved; [true] saves all seven; leaving
# the key out saves none.  k_f_<terr> and perfusion_p_<terr> are not part of
# this: they are always saved while either coupling is on.
TERRITORY_FIELDS = ('J3', 'J4', 'k_3', 'k_4_0', 'k_4_1', 'k_1', 'active_stress')


def _is_root(parent):
    """True on MPI rank 0, so start-up messages print once, not once per rank."""
    comm = getattr(parent, 'comm', None)
    return comm is None or comm.Get_rank() == 0


def _scalar(v):
    """JSON values may be written bare or MyoFE-style as one-element lists."""
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise ValueError("expected one value, got %s" % (v,))
        return v[0]
    return v


# The mechanics work in Pa.  Same factor circulation.py uses on the cavity
# pressure, which comes out of the same mixed function space.
PA_TO_MMHG = 0.0075

# MyoFE compiles forms with quadrature representation by default, for the
# Gauss-point myosim machinery.  Assembling these scalars without forcing
# uflacs can fail outright.  Same parameter LV_simulation.py passes on its
# myocardium_vol assembly -- keep them consistent.
FCP = {"representation": "uflacs"}

# NOTE on representation: total_stress contains cb_stress on a Quadrature
# element, which looks like it should need quadrature representation.  It
# does not.  LV_simulation.py already projects
#     inner(f0, total_stress*f0)
# with uflacs every time it writes mesh output, and that works.  Quadrature
# representation expands the Guccione exponentials symbolically at every
# quadrature point, and FFC takes effectively forever to compile it.  Use
# uflacs, matching what the rest of the codebase does.




class perfusion(object):

    def __init__(self, perfusion_struct, parent, initial_pressure_arteries):

        self.parent = parent
        self.data = {}
        self.model = {}

        for k in perfusion_struct.keys():
            self.model[k] = perfusion_struct[k][0]

        self.terminal_resistance = self.model.get('terminal_resistance', None)

        # ------------------------------------------------------------------
        # DEFAULTS = the production configuration.  An empty JSON block,
        #     "perfusion": {}
        # runs the full 16-segment R-L-C tree, Cauchy-stress IMP (Eq. 17), and
        # every LV territory at 1.0 mL/min/g.  The options below exist only
        # for testing; none needs to appear in the JSON.
        # ------------------------------------------------------------------

        # Terminal resistances.
        #   "physiological"  (default) every LV territory at 1.0 mL/min/g,
        #                    the healthy resting PET value (Lassen et al.
        #                    2026); see TERMINAL_RESISTANCE_PHYSIO
        #   "published"      Wang Table 1 Z as printed, ~2x physiological --
        #                    only for comparing against the paper
        self.rt_set = self.model.get('rt_set', 'physiological')

        # Stenosis: {segment name: percent AREA reduction}, Wang Eq. 1.
        #     "stenosis": [{"LMCA": 80}]              one segment
        #     "stenosis": [{"LMCA": 50, "RCA": 70}]   several
        # Leave it out for no stenosis.  Terminal resistances are NOT
        # recalibrated, so the narrowing reduces flow.
        self.stenosis = self.model.get('stenosis', {}) or {}
        if not isinstance(self.stenosis, dict):
            raise ValueError('stenosis must look like "stenosis": [{"LMCA": 80}]')

        # Perfusion: {vessel: percent of its healthy flow}.  The simple way.
        #     "perfusion_percent": [{"LAD": 20}]   LAD carries 20% of normal
        # Turned into the narrowing that gives that flow in the healthy heart
        # (coronary_rc.narrowing_for_flow), then applied like a stenosis.
        self.perfusion_percent = self.model.get('perfusion_percent', {}) or {}
        if not isinstance(self.perfusion_percent, dict):
            raise ValueError('perfusion_percent must look like '
                             '"perfusion_percent": [{"LAD": 20}]')

        subtree = self.model.get('subtree', 'full')
        if subtree not in SUBTREES:
            raise ValueError("unknown coronary subtree '%s'; options are %s"
                             % (subtree, sorted(SUBTREES.keys())))
        self.subtree = subtree

        if self.rt_set not in ('published', 'physiological'):
            raise ValueError("rt_set must be 'published' or 'physiological', "
                             "got '%s'" % self.rt_set)
        if self.terminal_resistance is None and self.rt_set == 'physiological':
            if subtree not in TERMINAL_RESISTANCE_PHYSIO:
                raise ValueError("no physiological terminal resistances for "
                                 "subtree '%s'" % subtree)
            self.terminal_resistance = TERMINAL_RESISTANCE_PHYSIO[subtree]

        self.imp_source = self.model.get('imp_source', 'stress')
        if self.imp_source == 'mesh':
            self.imp_source = 'multiplier'
        if self.imp_source not in ('multiplier', 'stress', 'radial',
                                   'prescribed'):
            raise ValueError(
                "imp_source must be 'multiplier', 'stress', 'radial' or "
                "'prescribed', got '%s'" % self.imp_source)

        # Scale applied to whichever mesh field is chosen.  1.0 is the honest
        # value; anything else is a modelling choice that must be reported.
        self.imp_scale = float(self.model.get('imp_scale', 1.0))

        # Write BOTH candidate IMP fields to data.csv for comparison.  Only
        # imp_source drives the tree.  Costs an extra projection per step.
        self.imp_diagnostic = bool(self.model.get('imp_diagnostic', False))

        # Solve the full R-L-C system of Wang Eq. 3 rather than the R-C
        # reduction.  Segment flows become states alongside node pressures,
        # so the system doubles in size.  See coronary_rc.py for why the
        # difference is expected to be small.
        self.inertance = bool(self.model.get('inertance', True))

        # Geometry comes only from aha_segmentation.py -- no JSON override, so
        # perfusion can never segment the mesh differently from what was
        # validated.
        self.aha_params = dict(aha.AHA_GEOMETRY)

        # The protocol, and therefore the timestep, does not exist when
        # LV_simulation.__init__ runs.  A throwaway tree is built here only so
        # the territory names -- and hence the output column names -- are
        # known before create_data_structure runs.  The real one is built on
        # the first timestep.
        self.time_step = None

        # Which coupling, read now: autoregulation targets its Q_ref.
        given = [k for k in COUPLING_TYPES if k in self.model]
        if len(given) > 1:
            raise ValueError("perfusion: give attachment_coupling OR "
                             "detachment_coupling, not both")
        kind = given[0] if given else 'attachment_coupling'
        self.setup_autoregulation(self.model.get('autoregulation', None),
                                  self.model.get(kind, None))
        law = self.autoreg['law'] if self.autoreg is not None else None

        self.narrowing_from_percent = {}
        if self.perfusion_percent:
            self.narrowing_from_percent = narrowing_for_flow(
                subtree, self.perfusion_percent,
                terminal_resistance=self.terminal_resistance,
                inertance=self.inertance, fixed_stenosis=self.stenosis,
                autoregulation=law)
            merged = dict(self.stenosis)
            merged.update(dict((s, a) for s, a in
                               self.narrowing_from_percent.items() if a > 0.0))
            self.stenosis = merged
        self.tree = CoronaryRC(subtree, 1.0,
                               terminal_resistance=self.terminal_resistance,
                               inertance=self.inertance,
                               stenosis=self.stenosis,
                               check_units=not self.narrowing_from_percent)

        if _is_root(self.parent) and self.perfusion_percent:
            print("perfusion: perfusion_percent (share of healthy flow%s)"
                  % (", after full autoregulatory dilation" if law else ""))
            for name in sorted(self.perfusion_percent):
                print("    %-6s %5.1f%% of normal flow  ->  vessel narrowed "
                      "%.2f%% of its area"
                      % (name, float(self.perfusion_percent[name]),
                         self.narrowing_from_percent[name]))

        if self.autoreg is not None:
            # Start where autoregulation settles for this narrowing at the
            # reference pressures, so the run has no start-up transient.
            a = self.autoreg
            a['R'] = autoregulated_state(
                self.tree, law['q_target'], law['r_min'], law['r_max'],
                REF_MEAN_P_AO, REF_MEAN_IMP)
            for s in self.tree.terminals:
                self.data['R_over_R0_' + s] = a['R'][s] / a['R0'][s]
            if _is_root(self.parent):
                print("perfusion: autoregulation starts adapted, R/R0 "
                      "(* = fully dilated)  " + "  ".join(
                          "%s %.3f%s" % (s, a['R'][s] / a['R0'][s],
                                         self._dilated_mark(s))
                          for s in self.tree.terminals))

        if not _is_root(self.parent):
            pass
        elif self.tree.stenosis:
            print("perfusion: stenosis (percent AREA reduction, Wang Eq. 1)")
            for name in sorted(self.tree.stenosis):
                a = self.tree.stenosis[name]
                print("    %-6s %5.1f%%  beta %.3f   R %.4f -> %.4f   "
                      "L %.4f -> %.4f   C %.3e -> %.3e"
                      % (name, a['percent'], a['beta'],
                         a['before'][0], a['after'][0],
                         a['before'][1], a['after'][1],
                         a['before'][2], a['after'][2]))
        else:
            print("perfusion: no stenosis")

        self.imp_peak_mmHg = self.model.get('imp_peak', {})
        for s in self.tree.terminals:
            if s not in self.imp_peak_mmHg:
                self.imp_peak_mmHg[s] = 70.0

        self.systole_fraction = self.model.get('systole_fraction', 0.4375)

        self.P = None
        self.initial_pressure_arteries = initial_pressure_arteries

        # territory marker, built lazily on the first mesh-sourced call
        self.dx = None
        self.territory_volume = None
        self._V0 = None

        self.data['coronary_P_AO'] = initial_pressure_arteries
        for s in self.tree.terminals:
            self.data['coronary_flow_' + s] = 0.0
            self.data['coronary_imp_' + s] = 0.0
        # Every segment's flow, conduits included.  A conduit carries the sum
        # of its children and has no perfusion territory, so it gets a
        # coronary_seg_* column but no IMP.  Node pressures too: these are
        # what the FFR calculation needs for the stenosis work.
        for s in self.tree.names:
            self.data['coronary_seg_' + s] = 0.0
        for n in self.tree.free_nodes:
            self.data['coronary_P_' + n] = 0.0
        if self.imp_diagnostic:
            for s in self.tree.terminals:
                self.data['imp_multiplier_' + s] = 0.0
                self.data['imp_stress_' + s] = 0.0
                self.data['imp_radial_' + s] = 0.0

        # Territory of every local integration point and its volume weight,
        # built once on the first step (build_point_map).  Shared by the
        # perfusion coupling and the territory outputs.
        self.point_term = None
        self.point_w = None

        try:
            self.setup_coupling(self.model.get(kind, None), kind)
        except ValueError as e:      # messages are worded for detachment
            raise ValueError(str(e).replace('detachment_coupling', kind))
        self.setup_territory_outputs(self.model.get('territory_outputs', None))

    # -----------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------

    def build(self, time_step):
        """Factorize the coronary matrix for this timestep and set the initial
        state.  Deferred to the first timestep because the protocol, and so
        dt, does not exist at construction."""
        self.time_step = time_step
        self.tree = CoronaryRC(self.subtree, time_step,
                               terminal_resistance=self.terminal_resistance,
                               inertance=self.inertance,
                               stenosis=self.stenosis,
                               check_units=not self.narrowing_from_percent)
        if self.autoreg is not None:
            self.tree.set_terminal_resistance(self.autoreg['R'])
        self.P = self.tree.steady_state(self.initial_pressure_arteries,
                                        self.return_prescribed_imp(0.0))

    def setup_autoregulation(self, spec, coupling_spec=None):
        """Parse "autoregulation".  Left out = ON with the defaults; false =
        off.  Sets self.autoreg.

        coupling_spec is the coupling's own block, read only for a q_ref
        override, so autoregulation and the coupling share one reference."""
        self.autoreg = None
        if spec is None and self.subtree != 'full':
            spec = False             # only set up for the full tree
        if spec is False:
            if _is_root(self.parent):
                print("perfusion: autoregulation OFF (terminal resistances "
                      "fixed)")
            return
        if spec is None or spec is True:
            spec = {}
        if not isinstance(spec, dict):
            raise ValueError('autoregulation must look like '
                             '"autoregulation": [{"cfr": 3.3, "n_ar": 5}]')
        bad = [k for k in spec if k not in AUTOREG_KEYS]
        if bad:
            raise ValueError("autoregulation: unknown key(s) %s; allowed keys "
                             "are %s" % (sorted(bad), list(AUTOREG_KEYS)))
        if self.subtree != 'full':
            raise ValueError("autoregulation is set up for the full tree only "
                             "(its targets and reference pressures are)")

        cfr = float(_scalar(spec.get('cfr', 3.3)))
        n_ar = float(_scalar(spec.get('n_ar', 5)))
        p_upper = float(_scalar(spec.get('p_upper', AUTOREG_P_UPPER)))
        t_start = float(_scalar(spec.get('t_start_s', 0.0)))
        if n_ar < 1.0:
            raise ValueError("autoregulation: n_ar = %g beats, must be >= 1"
                             % n_ar)
        if t_start < 0.0:
            raise ValueError("autoregulation: t_start_s must be >= 0")

        tmp = CoronaryRC(self.subtree, 1.0,
                         terminal_resistance=self.terminal_resistance,
                         inertance=self.inertance)
        R0 = dict(tmp.Rt)
        r_min, r_max = autoregulation_bounds(R0, cfr, p_upper)

        # Targets, mL/s (the tree's unit): the coupling's Q_ref for the LV
        # territories, the healthy flow at the reference pressures for the
        # rest (PLA).
        q_ref = dict(DEFAULT_Q_REF)
        if isinstance(coupling_spec, dict) and 'q_ref' in coupling_spec:
            user = _scalar(coupling_spec['q_ref'])
            if isinstance(user, dict):        # setup_coupling checks it fully
                for k, v in user.items():
                    q_ref[k] = float(_scalar(v))
        P = tmp.steady_state(REF_MEAN_P_AO, REF_MEAN_IMP)
        q_healthy = tmp.perfusion(P, REF_MEAN_P_AO, REF_MEAN_IMP)
        q_target = {}
        for s in tmp.terminals:
            q_target[s] = q_ref[s] / 60.0 if s in q_ref else q_healthy[s]
            if not q_target[s] > 0.0:
                raise ValueError("autoregulation: target flow for %s must be "
                                 "positive" % s)

        self.autoreg = dict(
            cfr=cfr, n_ar=n_ar, p_upper=p_upper, t_start=t_start,
            R0=R0, R=dict(R0),
            law=dict(q_target=q_target, r_min=r_min, r_max=r_max),
            vol=dict((s, 0.0) for s in R0), T=0.0,
            last_t_RR=None, armed=False, n_beats=0)
        for s in tmp.terminals:
            self.data['R_over_R0_' + s] = 1.0

        if _is_root(self.parent):
            print("perfusion: autoregulation ON   CFR %g (fully dilated at "
                  "R0/%g), flow held up to %g mmHg, n_ar = %g beats, from "
                  "t = %g s" % (cfr, cfr, p_upper, n_ar, t_start))
            print("perfusion:   lower limit (mmHg) at which each bed is fully "
                  "dilated  " + "  ".join(
                      "%s %.1f" % (s, REF_MEAN_IMP[s] + (REF_MEAN_P_AO -
                                                         REF_MEAN_IMP[s]) / cfr)
                      for s in tmp.terminals))

    def _dilated_mark(self, s):
        a = self.autoreg
        return '*' if a['R'][s] <= a['law']['r_min'][s] * (1.0 + 1e-9) else ''

    def autoregulation_step(self, q, time_step, t):
        """Every step, from implement_time_step.  Beats are spotted from the
        heart-rate countdown exactly as coupling_step does.  On the step that
        starts a beat, the beat that ended moves every terminal resistance
        1/n_ar of the way to the value that would have given its target, and
        the tree is refactorized: the new resistances act from the next
        step.  Every rank runs the same tree on the same global IMP, so every
        rank computes the same resistances."""
        a = self.autoreg
        t_rr = self.parent.hr.data['t_RR']
        new_beat = a['last_t_RR'] is not None and t_rr > a['last_t_RR']
        a['last_t_RR'] = t_rr

        if new_beat and t >= a['t_start']:
            if a['armed'] and a['T'] > 0.0:
                qbar = dict((s, a['vol'][s] / a['T']) for s in a['vol'])
                law = a['law']
                a['R'] = autoregulation_update(a['R'], qbar, law['q_target'],
                                               law['r_min'], law['r_max'],
                                               a['n_ar'])
                self.tree.set_terminal_resistance(a['R'])
                a['n_beats'] += 1
                if _is_root(self.parent):
                    print("autoregulation: beat %d closed at t = %.3f s   "
                          "R/R0  %s" % (a['n_beats'], t, "  ".join(
                              "%s %.3f%s" % (s, a['R'][s] / a['R0'][s],
                                             self._dilated_mark(s))
                              for s in self.tree.terminals)))
            a['armed'] = True
            a['T'] = 0.0
            for s in a['vol']:
                a['vol'][s] = 0.0

        if a['armed']:
            for s in a['vol']:
                a['vol'][s] += q[s] * time_step
            a['T'] += time_step
        for s in a['R']:
            self.data['R_over_R0_' + s] = a['R'][s] / a['R0'][s]

    def build_territory_marker(self, mesh_model):
        """Tag every cell with the coronary terminal that supplies it, and
        record each territory's reference volume.

        Cell midpoints, not quadrature points: dx() integrates over cell
        subdomains, so the marker has to be per cell either way, and a
        midpoint is exact for a tetrahedron without assuming anything about
        quadrature ordering.  This matches the element-level convention used
        for material assignment -- a cell belongs to one territory, so no
        constitutive or boundary discontinuity ever falls inside an element.

        Purely local: each rank labels its own cells.  The only global
        operation is inside assemble(), which handles it.
        """
        from dolfin import MeshFunction, Cell, Measure, assemble, Constant

        mesh = mesh_model['mesh']

        n_cells = mesh.num_cells()

        centroids = np.zeros((n_cells, 3))
        for i in range(n_cells):
            pt = Cell(mesh, i).midpoint()
            centroids[i, 0] = pt.x()
            centroids[i, 1] = pt.y()
            centroids[i, 2] = pt.z()

        lam, theta, n_clamped = aha.lambda_theta(centroids, self.aha_params)
        segment = aha.classify(lam, theta, self.aha_params)

        # A large clamp count means apex_point or axis_length is wrong, which
        # silently shifts every band boundary.  Cells legitimately sit a hair
        # outside [0,1] at the very apex and base, so a handful is expected.
        self.n_clamped = int(n_clamped)
        if n_clamped > 0.02 * max(n_cells, 1):
            print("perfusion: WARNING %d of %d cell centroids fell outside "
                  "lambda in [0,1]. Check apex_point and axis_length."
                  % (n_clamped, n_cells))

        # Terminals that perfuse tissue outside this mesh (PLA -> right
        # ventricle).  They own no AHA segments, get no marker tag, and their
        # IMP is taken as a fraction of the LV mean -- see return_computed_imp.
        self.extra_lv = dict(EXTRA_LV_TERMINALS.get(self.subtree, {}))
        for t in self.extra_lv:
            if t not in self.tree.terminals:
                raise RuntimeError("extra-LV terminal '%s' is not a terminal "
                                   "of subtree '%s'" % (t, self.subtree))

        # AHA segment -> terminal tag.
        #
        # For the full tree the map is aha_segmentation.PERFUSION_REGIONS --
        # the one validated in ParaView -- not a copy.  Its region names must
        # be exactly the tree's LV terminals, or the run stops here.
        # The reduced trees are roll-ups used only during development and keep
        # coronary_rc's map.
        #
        # The integer tag of each territory is aha_segmentation.REGION_NAME_TO_ID
        # itself -- not derived from the order of the coronary tree -- so the
        # marker here carries exactly the same region ids as region_of() and the
        # ParaView region field, by construction rather than by coincidence.
        if self.subtree == 'full':
            regions = aha.PERFUSION_REGIONS
            lv_terms = [t for t in self.tree.terminals if t not in self.extra_lv]
            if sorted(regions) != sorted(lv_terms):
                raise RuntimeError(
                    "aha_segmentation.PERFUSION_REGIONS %s does not match the "
                    "LV terminals of the coronary tree %s"
                    % (sorted(regions), sorted(lv_terms)))
            tag_of = dict(aha.REGION_NAME_TO_ID)
        else:
            regions = self.tree.regions
            tag_of = dict((t, j + 1)
                          for j, t in enumerate(self.tree.terminals))
        self.tag_of = tag_of

        seg_to_tag = {}
        for term in self.tree.terminals:
            for s in regions.get(term, []):
                seg_to_tag[s] = tag_of[term]

        missing = [s for s in range(1, 18) if s not in seg_to_tag]
        if missing:
            raise RuntimeError("AHA segments %s are not assigned to any "
                               "terminal of '%s'" % (missing, self.subtree))

        marker = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        marker_array = marker.array()
        for i in range(n_cells):
            marker_array[i] = seg_to_tag[segment[i]]
        # Kept for the perfusion coupling, which needs each integration
        # point's territory (see coupling_point_map).
        self.cell_tag = np.array(marker_array, dtype=int)

        self.marker = marker
        self.dx = Measure('dx', domain=mesh, subdomain_data=marker)

        # Reference volumes are constant -- dx is the reference measure -- so
        # they are assembled once here and only the numerator is assembled
        # per timestep.
        self.territory_volume = {}
        for term in self.tree.terminals:
            if term in self.extra_lv:
                continue          # no LV tissue, so no volume to assemble
            vol = assemble(Constant(1.0) * self.dx(self.tag_of[term]),
                           form_compiler_parameters=FCP)
            if vol <= 0.0:
                raise RuntimeError(
                    "territory '%s' (AHA %s) has zero volume: no cell centroid "
                    "landed in it. Check the AHA frame parameters."
                    % (term, regions.get(term, [])))
            self.territory_volume[term] = vol

        total = sum(self.territory_volume.values())
        print("perfusion: IMP source = '%s', scale = %g, %s, Rt set '%s'"
              % (self.imp_source, self.imp_scale,
                 "R-L-C (Eq. 3)" if self.inertance else "R-C", self.rt_set))
        print("perfusion: territory volumes (reference configuration)")
        for term in self.tree.terminals:
            if term in self.extra_lv:
                print("    %-6s %12s  outside the LV mesh, IMP = %.3f x LV mean"
                      % (term, "-", self.extra_lv[term]))
            else:
                print("    %-6s %12.6f  %5.1f%%"
                      % (term, self.territory_volume[term],
                         100.0 * self.territory_volume[term] / total))

    # -----------------------------------------------------------------
    # perfusion -> cross-bridge coupling (attachment or detachment)
    # -----------------------------------------------------------------

    def setup_coupling(self, spec, kind='detachment_coupling'):
        """Parse "attachment_coupling" or "detachment_coupling" and create
        its data columns.

        spec is None (coupling off) or a dict with any of c, n_tau,
        t_start_s, q_ref, p_prescribed.  Unknown keys stop the run, so a
        typo can never silently leave the coupling at its defaults.
        """
        self.coupling = None
        ctype = COUPLING_TYPES[kind]
        if spec is None:
            if _is_root(self.parent):
                print("perfusion: perfusion -> cross-bridge coupling OFF")
            return
        if not isinstance(spec, dict):
            raise ValueError('%s must look like "%s": [{"c": %g, "n_tau": 3}]'
                             % (kind, kind, ctype['c']))
        bad = [k for k in spec if k not in COUPLING_KEYS]
        if bad:
            raise ValueError("%s: unknown key(s) %s; allowed "
                             "keys are %s" % (kind, sorted(bad),
                                              list(COUPLING_KEYS)))

        c = float(_scalar(spec.get('c', ctype['c'])))
        autoreg_on = getattr(self, 'autoreg', None) is not None
        p_ref = _scalar(spec.get('p_reference',
                                 'healthy' if autoreg_on else 'open_artery'))
        if p_ref not in P_REFERENCES:
            raise ValueError("detachment_coupling: p_reference must be one of "
                             "%s, got '%s'" % (list(P_REFERENCES), p_ref))
        if autoreg_on and p_ref == 'open_artery':
            raise ValueError(
                "detachment_coupling: p_reference 'open_artery' cannot be used "
                "with autoregulation (which is on unless switched off) -- "
                "autoregulation already keeps healthy territories at Q_ref, "
                "and the open tree has none.  Either remove p_reference, or, "
                "to rerun an old case without autoregulation, add "
                "\"autoregulation\": [false] to the perfusion block.")
        n_tau = float(_scalar(spec.get('n_tau', 3)))
        t_start = float(_scalar(spec.get('t_start_s', 0.0)))
        if not 0.0 <= c <= 1.0:
            raise ValueError("detachment_coupling: c = %g, must be in [0, 1]"
                             % c)
        if n_tau < 1.0:
            raise ValueError("detachment_coupling: n_tau = %g beats, must be "
                             ">= 1" % n_tau)
        if t_start < 0.0:
            raise ValueError("detachment_coupling: t_start_s must be >= 0")

        extra = EXTRA_LV_TERMINALS.get(self.subtree, {})
        terms = [t for t in self.tree.terminals if t not in extra]

        q_ref = dict(DEFAULT_Q_REF) if self.subtree == 'full' else {}
        user = _scalar(spec['q_ref']) if 'q_ref' in spec else {}
        if not isinstance(user, dict):
            raise ValueError('detachment_coupling: q_ref must look like '
                             '"q_ref": {"LAD1": 27.36, ...} (mL/min)')
        for k, v in user.items():
            if k not in terms:
                raise ValueError("detachment_coupling: q_ref given for '%s', "
                                 "which is not an LV territory of this tree; "
                                 "territories are %s" % (k, terms))
            q_ref[k] = float(_scalar(v))
        p_set = {}
        user = _scalar(spec['p_prescribed']) if 'p_prescribed' in spec else {}
        if not isinstance(user, dict):
            raise ValueError('detachment_coupling: p_prescribed must look like '
                             '"p_prescribed": {"LAD1": 0.0}')
        for k, v in user.items():
            if k not in terms:
                raise ValueError("detachment_coupling: p_prescribed given for "
                                 "'%s', which is not an LV territory of this "
                                 "tree; territories are %s" % (k, terms))
            v = float(_scalar(v))
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    "detachment_coupling: p_prescribed for %s is %g.  Give the "
                    "relative perfusion as a FRACTION: 0 = no flow, 1 = normal "
                    "(so 0.3, not 30)" % (k, v))
            p_set[k] = v

        missing = [t for t in terms if t not in q_ref]
        if missing:
            raise ValueError("detachment_coupling: no reference flow for %s "
                             "-- give them in q_ref (mL/min)" % missing)
        for t in terms:
            if q_ref[t] <= 0.0:
                raise ValueError("detachment_coupling: q_ref for %s must be "
                                 "positive" % t)

        self.coupling = dict(
            c=c, n_tau=n_tau, t_start=t_start, terms=terms, q_ref=q_ref,
            p_set=p_set,          # territories whose p_i is fixed, not measured
            k_f=dict((t, 1.0) for t in terms),
            p=dict((t, 1.0) for t in terms),
            vol=dict((t, 0.0) for t in terms), T=0.0,
            p_ref=p_ref,
            pao_int=0.0,          # time integral of P_AO over the open beat
            imp_int=dict((t, 0.0) for t in self.tree.terminals),
            open_tree=None,       # the tree with every vessel open
            last_t_RR=None,       # heart-rate countdown at the last step
            armed=False,          # True once a whole beat has started
            point_term=None,      # territory of every local integration point
            kind=kind, rates=ctype['rates'], what=ctype['what'],
            base=None,            # {rate: base value at every local point}
            n_beats=0)
        for t in terms:
            self.data['perfusion_p_' + t] = 1.0
            self.data['k_f_' + t] = 1.0

        if _is_root(self.parent):
            print("perfusion: %s coupling ON   k_f = %.3g + %.3g p  scales %s,  "
                  "n_tau = %g beats,  from t = %g s"
                  % (ctype['what'], c, 1.0 - c, " and ".join(ctype['rates']),
                     n_tau, t_start))
            print("perfusion:   Q_ref (mL/min)  " + "  ".join(
                "%s %.2f" % (t, q_ref[t]) for t in terms))
            print("perfusion:   p measured against %s"
                  % ("the open-artery flow at the current pressures"
                     if p_ref == 'open_artery' else "healthy Q_ref"))
            if p_set:
                print("perfusion:   p PRESCRIBED (not measured)  " + "  ".join(
                    "%s %g" % (t, p_set[t]) for t in terms if t in p_set))

    def build_point_map(self, hs_objs_list, mesh_model):
        """Territory of every local integration point, via the quadrature
        dofmap, so no ordering of quadrature points is assumed.  Built once.

        hs_objs_list[j] is the half-sarcomere at local quadrature dof j: it
        is filled from, and written back to, the local vector of a function
        on quadrature_space.  Each cell's points take that cell's territory,
        the same per-cell labelling that IMP is integrated over.  Each point
        also gets an equal share of its cell's reference volume, the weight
        for the territory means (the degree-2 tetrahedral rule has equal
        weights).
        """
        if self.point_term is not None:
            return
        if self.dx is None:
            self.build_territory_marker(mesh_model)
        from dolfin import Cell

        mesh = mesh_model['mesh']
        name_of = dict((tag, name) for name, tag in self.tag_of.items())
        dm = mesh_model['function_spaces']['quadrature_space'].dofmap()
        n = len(hs_objs_list)
        point_tag = -np.ones(n, dtype=int)
        point_w = np.zeros(n)
        n_set = 0
        for cell in range(len(self.cell_tag)):
            dofs = [int(d) for d in dm.cell_dofs(cell)]
            share = Cell(mesh, cell).volume() / max(len(dofs), 1)
            for d in dofs:
                if 0 <= d < n:
                    point_tag[d] = self.cell_tag[cell]
                    point_w[d] = share
                    n_set += 1
        if np.any(point_tag < 0) or n_set != n:
            raise RuntimeError(
                "perfusion: %d of %d local integration points got no "
                "territory (%d assignments).  The quadrature dofmap does not "
                "match hs_objs_list." % (int(np.sum(point_tag < 0)), n, n_set))

        self.point_term = [name_of.get(tg) for tg in point_tag]
        self.point_w = point_w

    def coupling_point_map(self, hs_objs_list, mesh_model):
        """Which coupled territory each local point is in, and its base
        k_4_0 / k_4_1 (the values that k_f scales)."""
        self.build_point_map(hs_objs_list, mesh_model)
        cp = self.coupling
        cp['point_term'] = [s if s in cp['terms'] else None
                            for s in self.point_term]
        cp['base'] = dict((r, np.array([h.myof.data[r] for h in hs_objs_list]))
                          for r in cp['rates'])

    # -----------------------------------------------------------------
    # territory outputs (J4, rates, active stress) -> data.csv
    # -----------------------------------------------------------------

    def setup_territory_outputs(self, spec):
        """Read "territory_outputs" and create the chosen columns.  They have
        to exist now, before LV_simulation builds its data structure, to be
        written at all.

        spec is None (left out: save nothing), true / false (all or nothing),
        or a dict naming each field true or false.
        """
        self.terr_out = None
        usage = ('territory_outputs must look like "territory_outputs": '
                 '[{"J4": true, "active_stress": true}] (fields %s), or '
                 '[true] for all of them' % ", ".join(TERRITORY_FIELDS))
        if spec is None:
            fields = []
        elif isinstance(spec, bool):
            fields = list(TERRITORY_FIELDS) if spec else []
        elif isinstance(spec, dict):
            bad = [k for k in spec if k not in TERRITORY_FIELDS]
            if bad:
                raise ValueError("territory_outputs: unknown field(s) %s.  %s"
                                 % (sorted(bad), usage))
            for k, v in spec.items():
                if not isinstance(_scalar(v), bool):
                    raise ValueError("territory_outputs: %s must be true or "
                                     "false.  %s" % (k, usage))
            fields = [f for f in TERRITORY_FIELDS if _scalar(spec.get(f, False))]
        else:
            raise ValueError(usage)

        if not fields:
            if _is_root(self.parent):
                print("perfusion: territory outputs OFF")
            return
        extra = EXTRA_LV_TERMINALS.get(self.subtree, {})
        terms = [t for t in self.tree.terminals if t not in extra]
        self.terr_out = dict(terms=terms, fields=fields, point_idx=None)
        for t in terms:
            for f in fields:
                self.data[f + '_' + t] = 0.0
        if _is_root(self.parent):
            print("perfusion: territory outputs ON  (%s per LV territory, "
                  "volume-weighted)" % ", ".join(fields))

    def territory_output_step(self, mesh_model):
        """Every timestep, after MyoSim and the FE solve: territory means of
        J_3, J_4, the rates in use and active stress.  Collective under MPI --
        every rank calls it every step, so the Allreduce always matches."""
        to = self.terr_out
        hs_list = getattr(self.parent, 'hs_objs_list', None)
        if to is None or hs_list is None or mesh_model is None:
            return
        terms = to['terms']
        if to['point_idx'] is None:
            self.build_point_map(hs_list, mesh_model)
            idx = dict((s, i) for i, s in enumerate(terms))
            to['point_idx'] = np.array([idx.get(s, -1)
                                        for s in self.point_term], dtype=int)

        pidx = to['point_idx']
        pw = self.point_w
        cb = getattr(self.parent, 'cb_stress_list', None)
        if cb is not None and len(cb) != len(hs_list):
            cb = None

        # row 0: weight; then one row per TERRITORY_FIELDS entry, in order
        # J3, J4, k_3, k_4_0, k_4_1, k_1, active stress
        acc = np.zeros((1 + len(TERRITORY_FIELDS), len(terms)))
        for j, h in enumerate(hs_list):
            i = pidx[j]
            if i < 0:
                continue
            w = pw[j]
            md = h.myof.data
            acc[0, i] += w
            fl = getattr(h.myof, 'fluxes', None)
            if fl is not None:
                acc[1, i] += w * float(np.sum(fl['J_3']))
                acc[2, i] += w * float(np.sum(fl['J_4']))
            acc[3, i] += w * float(md['k_3'])
            acc[4, i] += w * float(md['k_4_0'])
            acc[5, i] += w * float(md['k_4_1'])
            acc[6, i] += w * float(md.get('k_1', np.nan))
            if cb is not None:
                acc[7, i] += w * float(cb[j])

        comm = getattr(self.parent, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            total = np.zeros_like(acc)
            comm.Allreduce(acc, total, op=MPI.SUM)
            acc = total

        for i, s in enumerate(terms):
            W = acc[0, i]
            if W <= 0.0:
                continue
            for r, f in enumerate(TERRITORY_FIELDS):
                if f in to['fields']:
                    self.data[f + '_' + s] = acc[r + 1, i] / W

    def coupling_step(self, q, time_step, t, mesh_model, P_AO=None,
                      P_IMP=None):
        """Run by implement_time_step every step; LV_simulation.py is not
        changed.

        A new beat is spotted from the heart-rate module: its countdown t_RR
        falls by one timestep each step and jumps back up to a full period
        when a beat starts -- the same signal the analysis scripts use to
        find beats.  On that step the beat that just ended is closed and the
        new rates are set.  MyoSim has already run this step, so they act
        from the next timestep: a 1 ms lag on a ~890 ms beat.  The flow of
        the step that starts the beat then goes into the new beat, so the
        beats partition the flow exactly.
        """
        cp = self.coupling
        t_rr = self.parent.hr.data['t_RR']
        new_beat = cp['last_t_RR'] is not None and t_rr > cp['last_t_RR']
        cp['last_t_RR'] = t_rr

        if new_beat:
            self.update_rates(self.parent.hs_objs_list, mesh_model, t)

        if cp['armed']:
            for s in cp['terms']:
                cp['vol'][s] += q[s] * time_step
            if P_AO is not None:
                cp['pao_int'] += P_AO * time_step
                for s in cp['imp_int']:
                    cp['imp_int'][s] += P_IMP[s] * time_step
            cp['T'] += time_step

    def open_artery_flow(self):
        """Mean flow (mL/min) each terminal would get over the beat that just
        ended if no vessel were narrowed: the open tree's steady solution at
        the beat-mean aortic pressure and IMPs.  The tree is linear, so this is
        exactly the beat-mean flow an open tree would have carried."""
        cp = self.coupling
        T = cp['T']
        pao = cp['pao_int'] / T
        imp = dict((s, cp['imp_int'][s] / T) for s in cp['imp_int'])
        if cp['open_tree'] is None:
            cp['open_tree'] = CoronaryRC(self.subtree, 1.0,
                                         terminal_resistance=self.terminal_resistance,
                                         inertance=self.inertance)
        tree = cp['open_tree']
        P = tree.steady_state(pao, imp)
        q = tree.perfusion(P, pao, imp)
        return dict((s, 60.0 * q[s]) for s in q)

    def update_rates(self, hs_objs_list, mesh_model, t):
        """Once per beat, from coupling_step.

        Closes the beat that just ended: its mean flow per territory gives
        p_i, which moves k_f,i one n_tau-th of the way to c + (1 - c) p_i.
        Then sets the coupled rates (k_3, or k_4_0 and k_4_1) at every point
        in each territory.  Every
        rank runs the same coronary tree on the same global IMP, so every
        rank computes the same k_f and no communication is needed.
        """
        cp = self.coupling
        if cp is None or t < cp['t_start']:
            return
        if mesh_model is None:
            raise ValueError("the %s coupling needs the mesh to find each "
                             "point's territory; imp_source 'prescribed' "
                             "does not provide it" % cp['what'])

        if cp['armed'] and cp['T'] > 0.0:
            q_open = self.open_artery_flow() if cp['p_ref'] == 'open_artery' \
                else None
            for s in cp['terms']:
                if s in cp['p_set']:
                    p = cp['p_set'][s]                        # fixed in JSON
                else:
                    qbar = 60.0 * cp['vol'][s] / cp['T']      # mL/min
                    if q_open is None:
                        ref = cp['q_ref'][s]
                    else:
                        ref = q_open[s]
                    # no driving pressure even with an open artery: the
                    # narrowing cannot be what limits flow, so p = 1
                    p = 1.0 if ref <= 1e-9 else min(1.0, max(0.0, qbar / ref))
                target = cp['c'] + (1.0 - cp['c']) * p
                cp['k_f'][s] += (target - cp['k_f'][s]) / cp['n_tau']
                cp['p'][s] = p
                self.data['perfusion_p_' + s] = p
                self.data['k_f_' + s] = cp['k_f'][s]
            cp['n_beats'] += 1
            if _is_root(self.parent):
                print("perfusion: beat %d closed at t = %.3f s   k_f  %s"
                      % (cp['n_beats'], t, "  ".join(
                          "%s %.4f" % (s, cp['k_f'][s]) for s in cp['terms'])))
        cp['armed'] = True
        cp['T'] = 0.0
        for s in cp['terms']:
            cp['vol'][s] = 0.0
        cp['pao_int'] = 0.0
        for s in cp['imp_int']:
            cp['imp_int'][s] = 0.0

        if cp['point_term'] is None:
            self.coupling_point_map(hs_objs_list, mesh_model)

        kf = cp['k_f']
        base = cp['base']
        for j, s in enumerate(cp['point_term']):
            if s is None:
                continue
            md = hs_objs_list[j].myof.data
            for r in cp['rates']:
                md[r] = base[r][j] * kf[s]

    # -----------------------------------------------------------------
    # intramyocardial pressure
    # -----------------------------------------------------------------

    def return_cycle_length(self):
        hr = self.parent.data['heart_rate']
        if hr <= 0.0:
            raise ValueError("heart rate must be positive to phase the "
                             "prescribed IMP waveform")
        return 60.0 / hr

    def return_prescribed_imp(self, t):
        """Placeholder: half-sine over systole, zero in diastole.  mmHg."""
        T = self.return_cycle_length()
        T_sys = self.systole_fraction * T
        phase = t - T * np.floor(t / T)
        if phase < T_sys:
            shape = np.sin(np.pi * phase / T_sys)
        else:
            shape = 0.0
        return dict((s, self.imp_peak_mmHg[s] * shape)
                    for s in self.tree.terminals)

    def return_imp_all(self, mesh_model):
        """Territory averages of BOTH candidate IMP fields, in mmHg.

        Returns {'multiplier': {...}, 'stress': {...}}.  Only the field named
        by imp_source is used to drive the coronary tree; the other is
        computed for comparison and written to data.csv only.  Computing both
        from the SAME mechanics state is the point -- it removes any question
        of comparing across runs at different loading.

        Costs one extra DG0 projection per timestep, so turn the diagnostic
        off (imp_diagnostic = false) for production runs.
        """
        out = {}
        for src in ('multiplier', 'stress', 'radial'):
            field, fcp = self.return_imp_function(mesh_model, source=src)
            d = {}
            num = 0.0
            den = 0.0
            for term in self.tree.terminals:
                if term in self.extra_lv:
                    continue
                integral = assemble(field * self.dx(self.tag_of[term]),
                                    form_compiler_parameters=fcp)
                d[term] = PA_TO_MMHG * integral / self.territory_volume[term]
                num += integral
                den += self.territory_volume[term]
            lv_mean = PA_TO_MMHG * num / den
            for term, frac in self.extra_lv.items():
                d[term] = frac * lv_mean
            out[src] = d
        return out

    def return_imp_function(self, mesh_model, source=None):
        """The scalar field to average, as a DG0 Function in Pa.

        The expression -tr(F S F^T)/(3J) contains the whole Guccione
        exponential PK2 plus the active term.  Integrating it directly over
        each territory would ask FFC to compile one large form PER TERRITORY.
        It is projected ONCE onto DG0 instead -- a single compile, cached
        after the first timestep -- and the per-territory integrals are then
        trivial forms over a simple Function.  DG0 also makes the projection
        cheap, since its mass matrix is diagonal.
        """
        from dolfin import project, tr, sqrt, inner, FunctionSpace

        if source is None:
            source = self.imp_source

        if source == 'multiplier':
            # p is CG1 on the mixed element -- no quadrature elements, so it
            # can be integrated directly with no projection at all.
            return mesh_model['uflforms'].parameters["pressure_variable"], FCP

        F = mesh_model['functions']['Fmat']
        S = mesh_model['functions']['total_stress']
        J = mesh_model['functions']['J']

        if source == 'stress':
            # Eq. 17: the hydrostatic part of the Cauchy stress.
            #   tr(sigma) = tr(F S F^T)/J,  and  P F^T = F S F^T
            # No minus sign -- that is how the paper writes it, and the
            # negated version gives IMP that is NEGATIVE during systole.
            expr = tr(F * S * F.T) / (3.0 * J)
        else:
            # 'radial': minus the radial component of the Cauchy stress.
            #
            # A vessel running through the wall is squeezed by the stress
            # acting ACROSS it, not by an average over three directions --
            # and averaging lets the tensile hoop and longitudinal stress
            # cancel the compressive radial one.  Equilibrium pins this
            # component: sigma_rr = -P_cavity at the endocardium and 0 at
            # the epicardium, so -sigma_rr inherits cavity pressure's shape
            # as well as its scale, rather than being a narrow spike.
            #
            # err is the referential radial unit vector, read from the mesh
            # (ellipsoidal/eR); push it forward and renormalise.
            err = mesh_model['functions']['err']
            v = F * err
            e_r = v / sqrt(inner(v, v))
            sigma = (1.0 / J) * F * S * F.T
            expr = -inner(e_r, sigma * e_r)

        if self._V0 is None:
            self._V0 = FunctionSpace(mesh_model['mesh'], 'DG', 0)

        return project(expr, self._V0,
                       form_compiler_parameters=FCP), FCP

    def return_computed_imp(self, mesh_model):
        """Volume-average the chosen field over each coronary territory.
        Returns mmHg keyed by terminal segment."""
        from dolfin import assemble

        field, fcp = self.return_imp_function(mesh_model)

        out = {}
        num = 0.0
        den = 0.0
        for term in self.tree.terminals:
            if term in self.extra_lv:
                continue
            integral = assemble(field * self.dx(self.tag_of[term]),
                                form_compiler_parameters=fcp)
            out[term] = (self.imp_scale * PA_TO_MMHG * integral
                         / self.territory_volume[term])
            num += integral
            den += self.territory_volume[term]

        # Terminals outside the LV mesh take a fraction of the volume-weighted
        # LV mean, following Wang et al.'s treatment of the PLA region.
        lv_mean = self.imp_scale * PA_TO_MMHG * num / den
        for term, frac in self.extra_lv.items():
            out[term] = frac * lv_mean
        return out

    # -----------------------------------------------------------------
    # time stepping
    # -----------------------------------------------------------------

    def implement_time_step(self, pressure_arteries, time_step, t,
                            mesh_model=None):
        """Advance the coronary tree one step.

        pressure_arteries is mmHg, straight from circ.data['pressure_aorta'],
        and is used as-is: the tree runs in the same units.

        mesh_model is self.mesh.model -- the dict holding the mesh, the
        function dictionary and uflforms.  Required unless imp_source is
        'prescribed'.  Call this AFTER the mechanics solve, or the fields
        will be one timestep stale.
        """
        if self.time_step is None:
            self.build(time_step)
        elif abs(time_step - self.time_step) > 1e-12:
            raise ValueError(
                "coronary matrix was factorized for dt=%g but the simulation "
                "is stepping at dt=%g; rebuild the tree if dt changes"
                % (self.time_step, time_step))

        if self.imp_source == 'prescribed':
            P_IMP = self.return_prescribed_imp(t)
        else:
            if mesh_model is None:
                raise ValueError(
                    "imp_source is '%s' but implement_time_step was called "
                    "without mesh_model" % self.imp_source)
            if self.dx is None:
                self.build_territory_marker(mesh_model)
            if self.imp_diagnostic:
                both = self.return_imp_all(mesh_model)
                for s in self.tree.terminals:
                    self.data['imp_multiplier_' + s] = both['multiplier'][s]
                    self.data['imp_stress_' + s] = both['stress'][s]
                    self.data['imp_radial_' + s] = both['radial'][s]
                P_IMP = dict((s, self.imp_scale * both[self.imp_source][s])
                             for s in self.tree.terminals)
            else:
                P_IMP = self.return_computed_imp(mesh_model)

        self.P = self.tree.step(self.P, pressure_arteries, P_IMP)

        q = self.tree.perfusion(self.P, pressure_arteries, P_IMP)

        self.data['coronary_P_AO'] = pressure_arteries
        for s in self.tree.terminals:
            self.data['coronary_flow_' + s] = q[s]
            self.data['coronary_imp_' + s] = P_IMP[s]

        if self.coupling is not None:
            self.coupling_step(q, time_step, t, mesh_model,
                               P_AO=pressure_arteries, P_IMP=P_IMP)
        if self.autoreg is not None:
            self.autoregulation_step(q, time_step, t)
        if self.terr_out is not None:
            self.territory_output_step(mesh_model)

        seg = self.tree.flows(self.P, pressure_arteries, P_IMP)
        for s in self.tree.names:
            self.data['coronary_seg_' + s] = seg[s]
        for n in self.tree.free_nodes:
            self.data['coronary_P_' + n] = self.P[self.tree.idx[n]]

        return q
