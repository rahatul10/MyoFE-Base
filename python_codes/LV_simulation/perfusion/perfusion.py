"""
Perfusion module: drives the 0D coronary tree from the running simulation.

Stage 2 -- one-way coupling only.

    P_AO  <- taken from the systemic circulation each timestep (real)
    P_IMP <- prescribed analytic waveform (placeholder for Eq. 17)

Nothing computed here feeds back into the mechanics, so this cannot
destabilise a simulation.  Eq. 17 (real IMP) and Eqs. 19-20 (eta) come later.

The configuration is chosen in the JSON with "subtree".  Territories,
terminal resistances and the exposed data columns all follow from it, so
growing the tree needs no change in this file.

UNITS
-----
Everything is mmHg.  MyoFE's circulation works in mmHg (see circulation.py,
the 0.0075 factor converting LV cavity pressure from Pa) and the coronary
tree now does too, so there is no conversion anywhere in this module.

Wang et al.'s Table 1 is labelled kPa s cm^-3, but the values are almost
certainly mmHg s mL^-1 -- see the header of coronary_rc.py for the evidence.
An earlier version of this file divided P_AO by 7.50062 to convert to kPa
while the resistances were already in mmHg, a 7.5x inconsistency that the
terminal-resistance calibration silently absorbed.

Python 2.7 compatible.
"""

import numpy as np

from .coronary_rc import CoronaryRC, SUBTREES


class perfusion(object):

    def __init__(self, perfusion_struct, parent, initial_pressure_arteries):

        self.parent = parent
        self.data = {}
        self.model = {}

        for k in perfusion_struct.keys():
            self.model[k] = perfusion_struct[k][0]

        # Downstream resistance replacing the microcirculation removed by
        # truncating the tree.  None means use the calibrated table for this
        # configuration (see coronary_rc.TERMINAL_RESISTANCE); JSON may
        # override with a dict keyed by territory.
        self.terminal_resistance = self.model.get('terminal_resistance', None)

        subtree = self.model.get('subtree', 'lad_lcx_rca')
        if subtree not in SUBTREES:
            raise ValueError("unknown coronary subtree '%s'; options are %s"
                             % (subtree, sorted(SUBTREES.keys())))

        # The protocol (and therefore the timestep) does not exist yet when
        # LV_simulation.__init__ runs -- self.prot is created later, inside
        # run_simulation().  The tree is therefore built lazily on the first
        # call to implement_time_step, when dt is known.  A throwaway tree is
        # built here only so the territory names, and hence the output column
        # names, are known before create_data_structure runs.
        self.subtree = subtree
        self.time_step = None
        self.tree = CoronaryRC(
            subtree,
            1.0,
            terminal_resistance=self.terminal_resistance)

        # prescribed IMP placeholder, peak values in mmHg per territory
        self.imp_peak_mmHg = self.model.get('imp_peak', {})
        for s in self.tree.terminals:
            if s not in self.imp_peak_mmHg:
                self.imp_peak_mmHg[s] = 70.0

        self.systole_fraction = self.model.get('systole_fraction', 0.4375)

        self.P = None
        self.initial_pressure_arteries = initial_pressure_arteries

        self.data['coronary_P_AO'] = initial_pressure_arteries
        for s in self.tree.terminals:
            self.data['coronary_flow_' + s] = 0.0
            self.data['coronary_imp_' + s] = 0.0

    def build(self, time_step):
        """Factorize the coronary matrix for this timestep and set the
        initial state.  Called on the first timestep, not at construction,
        because the protocol does not exist until run_simulation()."""
        self.time_step = time_step
        self.tree = CoronaryRC(
            self.subtree,
            time_step,
            terminal_resistance=self.terminal_resistance)
        self.P = self.tree.steady_state(self.initial_pressure_arteries,
                                        self.return_prescribed_imp(0.0))

    def return_cycle_length(self):
        hr = self.parent.data['heart_rate']
        if hr <= 0.0:
            raise ValueError("heart rate must be positive to phase the "
                             "prescribed IMP waveform")
        return 60.0 / hr

    def return_prescribed_imp(self, t):
        """Placeholder for Eq. 17.  Half-sine over systole, zero in diastole.
        Returns mmHg keyed by terminal segment."""
        T = self.return_cycle_length()
        T_sys = self.systole_fraction * T
        phase = t - T * np.floor(t / T)
        if phase < T_sys:
            shape = np.sin(np.pi * phase / T_sys)
        else:
            shape = 0.0
        return dict((s, self.imp_peak_mmHg[s] * shape)
                    for s in self.tree.terminals)

    def implement_time_step(self, pressure_arteries, time_step, t):
        """Advance the coronary tree one step.

        pressure_aorta is mmHg, straight from self.circ.data, and is used
        as-is: the coronary tree runs in the same units.
        """
        if self.time_step is None:
            self.build(time_step)
        elif abs(time_step - self.time_step) > 1e-12:
            raise ValueError(
                "coronary matrix was factorized for dt=%g but the simulation "
                "is stepping at dt=%g; rebuild the tree if dt changes"
                % (self.time_step, time_step))

        P_IMP = self.return_prescribed_imp(t)

        self.P = self.tree.step(self.P, pressure_arteries, P_IMP)

        q = self.tree.perfusion(self.P, pressure_arteries, P_IMP)

        self.data['coronary_P_AO'] = pressure_arteries
        for s in self.tree.terminals:
            self.data['coronary_flow_' + s] = q[s]
            self.data['coronary_imp_' + s] = P_IMP[s]

        return q
