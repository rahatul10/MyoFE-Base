"""
Perfusion module: drives the 0D coronary tree from the running simulation.

Stage 2 -- one-way coupling only.

    P_AO  <- taken from the systemic circulation each timestep (real)
    P_IMP <- prescribed analytic waveform (placeholder for Eq. 17)

Nothing computed here feeds back into the mechanics, so this cannot
destabilise a simulation.  Eq. 17 (real IMP) and Eqs. 19-20 (eta) come later.

UNITS
-----
MyoFE's circulation works in mmHg (see circulation.py, the 0.0075 factor
converting LV cavity pressure from Pa).  The coronary tree follows the paper
and works in kPa.  Every exchange across this boundary is converted here and
nowhere else.

Python 2.7 compatible.
"""

import numpy as np

from .coronary_rc import (CoronaryRC, SUBTREES, MMHG_PER_KPA,
                          TERMINAL_RESISTANCE)


class perfusion(object):

    def __init__(self, perfusion_struct, parent, initial_pressure_arteries):

        self.parent = parent
        self.data = {}
        self.model = {}

        for k in perfusion_struct.keys():
            self.model[k] = perfusion_struct[k][0]

        # Downstream resistance replacing the microcirculation removed by
        # truncating the tree.  JSON may override; default is calibrated to
        # measured resting flows (see coronary_rc.TERMINAL_RESISTANCE).
        self.terminal_resistance = self.model.get('terminal_resistance',
                                                  TERMINAL_RESISTANCE)

        subtree = self.model.get('subtree', 'lmca_rca')
        if subtree not in SUBTREES:
            raise ValueError("unknown coronary subtree '%s'; options are %s"
                             % (subtree, sorted(SUBTREES.keys())))

        # The protocol (and therefore the timestep) does not exist yet when
        # LV_simulation.__init__ runs -- self.prot is created later, inside
        # run_simulation().  The tree is therefore built lazily on the first
        # call to implement_time_step, when dt is known.
        self.subtree = subtree
        self.time_step = None
        self.tree = CoronaryRC(
            SUBTREES[subtree],
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
            SUBTREES[self.subtree],
            time_step,
            terminal_resistance=self.terminal_resistance)
        P_AO_kPa = self.initial_pressure_arteries / MMHG_PER_KPA
        self.P = self.tree.steady_state(P_AO_kPa,
                                        self.return_prescribed_imp(0.0))

    def return_cycle_length(self):
        hr = self.parent.data['heart_rate']
        if hr <= 0.0:
            raise ValueError("heart rate must be positive to phase the "
                             "prescribed IMP waveform")
        return 60.0 / hr

    def return_prescribed_imp(self, t):
        """Placeholder for Eq. 17.  Half-sine over systole, zero in diastole.
        Returns kPa keyed by terminal segment."""
        T = self.return_cycle_length()
        T_sys = self.systole_fraction * T
        phase = t - T * np.floor(t / T)
        if phase < T_sys:
            shape = np.sin(np.pi * phase / T_sys)
        else:
            shape = 0.0
        return dict((s, self.imp_peak_mmHg[s] * shape / MMHG_PER_KPA)
                    for s in self.tree.terminals)

    def implement_time_step(self, pressure_arteries, time_step, t):
        """Advance the coronary tree one step.

        pressure_arteries is mmHg, straight from self.circ.data.
        """
        if self.time_step is None:
            self.build(time_step)
        elif abs(time_step - self.time_step) > 1e-12:
            raise ValueError(
                "coronary matrix was factorized for dt=%g but the simulation "
                "is stepping at dt=%g; rebuild the tree if dt changes"
                % (self.time_step, time_step))

        P_AO_kPa = pressure_arteries / MMHG_PER_KPA
        P_IMP_kPa = self.return_prescribed_imp(t)

        self.P = self.tree.step(self.P, P_AO_kPa, P_IMP_kPa)

        q = self.tree.perfusion(self.P, P_AO_kPa, P_IMP_kPa)

        self.data['coronary_P_AO'] = pressure_arteries
        for s in self.tree.terminals:
            self.data['coronary_flow_' + s] = q[s]
            self.data['coronary_imp_' + s] = P_IMP_kPa[s] * MMHG_PER_KPA

        return q
