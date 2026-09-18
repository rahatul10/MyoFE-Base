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

from .coronary_rc import CoronaryRC, SUBTREES, EXTRA_LV_TERMINALS


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


# Geometry of the AHA frame.  Measured offline in serial, because under MPI
# no rank sees both the apex and the base.  Overridable from JSON; these
# defaults are the values recorded in dependencies/aha_segmentation.py for
# the baseline mesh.  If the mesh changes, these must be re-measured.
DEFAULT_AHA = {
    "apex_point":  [0.0, 0.0, -0.7315],
    "axis_vector": [0.0, 0.0, 1.0],
    "axis_length": 0.7315,
    "c_ref":       [1.0, 0.0, 0.0],
    "lambda_c":    0.0902,
    "lambda_A":    0.3935,
    "lambda_M":    0.6967,
}


class perfusion(object):

    def __init__(self, perfusion_struct, parent, initial_pressure_arteries):

        self.parent = parent
        self.data = {}
        self.model = {}

        for k in perfusion_struct.keys():
            self.model[k] = perfusion_struct[k][0]

        self.terminal_resistance = self.model.get('terminal_resistance', None)

        subtree = self.model.get('subtree', 'lad_lcx_rca')
        if subtree not in SUBTREES:
            raise ValueError("unknown coronary subtree '%s'; options are %s"
                             % (subtree, sorted(SUBTREES.keys())))
        self.subtree = subtree

        self.imp_source = self.model.get('imp_source', 'prescribed')
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
        self.inertance = bool(self.model.get('inertance', False))

        self.aha_params = dict(DEFAULT_AHA)
        self.aha_params.update(self.model.get('aha', {}))

        # The protocol, and therefore the timestep, does not exist when
        # LV_simulation.__init__ runs.  A throwaway tree is built here only so
        # the territory names -- and hence the output column names -- are
        # known before create_data_structure runs.  The real one is built on
        # the first timestep.
        self.time_step = None
        self.tree = CoronaryRC(subtree, 1.0,
                               terminal_resistance=self.terminal_resistance,
                               inertance=self.inertance)

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
        if self.imp_diagnostic:
            for s in self.tree.terminals:
                self.data['imp_multiplier_' + s] = 0.0
                self.data['imp_stress_' + s] = 0.0
                self.data['imp_radial_' + s] = 0.0

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
                               inertance=self.inertance)
        self.P = self.tree.steady_state(self.initial_pressure_arteries,
                                        self.return_prescribed_imp(0.0))

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

        try:
            from ..dependencies import aha_segmentation as aha
        except ImportError:
            import aha_segmentation as aha

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

        # AHA segment -> terminal tag, via this configuration's territory map
        seg_to_tag = {}
        for j, term in enumerate(self.tree.terminals):
            for s in self.tree.regions[term]:
                seg_to_tag[s] = j + 1

        missing = [s for s in range(1, 18) if s not in seg_to_tag]
        if missing:
            raise RuntimeError("AHA segments %s are not assigned to any "
                               "terminal of '%s'" % (missing, self.subtree))

        marker = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        marker_array = marker.array()
        for i in range(n_cells):
            marker_array[i] = seg_to_tag[segment[i]]

        self.marker = marker
        self.dx = Measure('dx', domain=mesh, subdomain_data=marker)

        # Reference volumes are constant -- dx is the reference measure -- so
        # they are assembled once here and only the numerator is assembled
        # per timestep.
        self.territory_volume = {}
        for j, term in enumerate(self.tree.terminals):
            if term in self.extra_lv:
                continue          # no LV tissue, so no volume to assemble
            vol = assemble(Constant(1.0) * self.dx(j + 1),
                           form_compiler_parameters=FCP)
            if vol <= 0.0:
                raise RuntimeError(
                    "territory '%s' (AHA %s) has zero volume: no cell centroid "
                    "landed in it. Check the AHA frame parameters."
                    % (term, self.tree.regions[term]))
            self.territory_volume[term] = vol

        total = sum(self.territory_volume.values())
        print("perfusion: IMP source = '%s', scale = %g, %s"
              % (self.imp_source, self.imp_scale,
                 "R-L-C (Eq. 3)" if self.inertance else "R-C"))
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
            for j, term in enumerate(self.tree.terminals):
                if term in self.extra_lv:
                    continue
                integral = assemble(field * self.dx(j + 1),
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
        for j, term in enumerate(self.tree.terminals):
            if term in self.extra_lv:
                continue
            integral = assemble(field * self.dx(j + 1),
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

        return q
