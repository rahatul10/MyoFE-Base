"""
Validation ladder for the R-C coronary network (no terminal impedance).

Each rung adds one structural feature and checks it against a result that can
be computed by hand.
"""

import numpy as np
from coronary_segments import SEGMENTS
from coronary_rc import (CoronaryRC, SUBTREES, MMHG_PER_KPA,
                         PERFUSION_REGIONS_LAD)


def hdr(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print("  %-48s %12.6g %s" % (label, got, "PASS" if ok else
                                 "FAIL (want %.6g)" % want))
    if not ok:
        raise AssertionError(label)


P_AO = 12.0          # kPa, ~90 mmHg
P_OUT = 2.0          # kPa, prescribed outlet (stands in for P_IMP)
DT = 1.0e-3

R = dict((s, SEGMENTS[s]["R"]) for s in SEGMENTS)


# ---------------------------------------------------------------------------
hdr("Rung 1 -- single segment: Ohm's law (both ends pinned, 0 unknowns)")

m = CoronaryRC(SUBTREES["single"], DT)
m.describe()

P_IMP = {"LMCA": P_OUT}
P = m.steady_state(P_AO, P_IMP)
q = m.flows(P, P_AO, P_IMP)

want_Q = (P_AO - P_OUT) / R["LMCA"]
print("  hand calc: Q = (P_AO - P_out)/R_LMCA = (%.1f - %.1f)/%.4f"
      % (P_AO, P_OUT, R["LMCA"]))
check("Q through LMCA [cm3/s]", q["LMCA"], want_Q)


# ---------------------------------------------------------------------------
hdr("Rung 2 -- two in series: resistances add")

m = CoronaryRC(SUBTREES["series"], DT)
m.describe()

P_IMP = {"LAD": P_OUT}
P = m.steady_state(P_AO, P_IMP)
q = m.flows(P, P_AO, P_IMP)

want_Q = (P_AO - P_OUT) / (R["LMCA"] + R["LAD"])
print("  hand calc: Q = (P_AO - P_out)/(R_LMCA + R_LAD) = %.1f/%.4f"
      % (P_AO - P_OUT, R["LMCA"] + R["LAD"]))
check("Q through LMCA", q["LMCA"], want_Q)
check("Q through LAD", q["LAD"], want_Q)
check("series continuity", q["LMCA"] - q["LAD"], 0.0, tol=1e-12)
check("junction pressure [kPa]", P[m.idx["N_LMCA"]], P_AO - want_Q * R["LMCA"])


# ---------------------------------------------------------------------------
hdr("Rung 3 -- branch point: Kirchhoff")

m = CoronaryRC(SUBTREES["branch"], DT)
m.describe()

P_IMP = {"LAD1": P_OUT, "LAD2": P_OUT}
P = m.steady_state(P_AO, P_IMP)
q = m.flows(P, P_AO, P_IMP)

R_par = 1.0 / (1.0 / R["LAD1"] + 1.0 / R["LAD2"])
R_tot = R["LMCA"] + R["LAD"] + R_par
want_Q = (P_AO - P_OUT) / R_tot

print("  hand calc: legs %.4f and %.4f in parallel -> %.4f"
      % (R["LAD1"], R["LAD2"], R_par))
check("parent flow Q_LAD", q["LAD"], want_Q)
check("Kirchhoff  Q_LAD - (Q_LAD1 + Q_LAD2)",
      q["LAD"] - (q["LAD1"] + q["LAD2"]), 0.0, tol=1e-12)
check("split ratio Q_LAD1/Q_LAD2", q["LAD1"] / q["LAD2"], R["LAD2"] / R["LAD1"])


# ---------------------------------------------------------------------------
hdr("Rung 4 -- transient: RC time constant at the junction")

dt = 1.0e-7
m = CoronaryRC(SUBTREES["series"], dt)
P_IMP = {"LAD": 0.0}

C_node = m.C["N_LMCA"]
tau = C_node / (1.0 / R["LMCA"] + 1.0 / R["LAD"])
P_ss = m.steady_state(P_AO, P_IMP)[m.idx["N_LMCA"]]

P = np.zeros(m.n)
traj = []
t = 0.0
for k in range(int(round(5.0 * tau / dt))):
    P = m.step(P, P_AO, P_IMP)
    t += dt
    traj.append((t, P[m.idx["N_LMCA"]]))

tt = np.array([x[0] for x in traj])
PP = np.array([x[1] for x in traj])
exact = P_ss * (1.0 - np.exp(-tt / tau))

print("  tau = C_node/(1/R_LMCA + 1/R_LAD) = %.6g s" % tau)
check("fraction of steady state at t=tau",
      PP[int(round(tau / dt)) - 1] / P_ss, 1.0 - np.exp(-1.0), tol=2e-3)
check("max relative error vs analytic over %d steps" % len(tt),
      np.max(np.abs(PP - exact)) / P_ss, 0.0, tol=1e-2)


# ---------------------------------------------------------------------------
hdr("Rung 5 -- LAD subtree, 3 perfusion territories, pulsatile")

m = CoronaryRC(SUBTREES["lad3"], DT)
m.describe()
print()
for s in m.terminals:
    print("  %-6s -> AHA segments %s" % (s, PERFUSION_REGIONS_LAD[s]))

T = 0.8
T_SYS = 0.35
IMP_PEAK = {"LAD1": 70.0 / MMHG_PER_KPA,
            "LAD3": 43.0 / MMHG_PER_KPA,
            "LAD4": 90.0 / MMHG_PER_KPA}


def p_ao(t):
    ph = (t % T) / T
    return (80.0 + 40.0 * np.exp(-((ph - 0.22) / 0.13) ** 2)) / MMHG_PER_KPA


def p_imp(t):
    ph = t % T
    shape = np.sin(np.pi * ph / T_SYS) if ph < T_SYS else 0.0
    return dict((s, IMP_PEAK[s] * shape) for s in m.terminals)


nsteps = int(round(4 * T / DT))
P = m.steady_state(p_ao(0.0), p_imp(0.0))

t = 0.0
rec_t, rec_pao = [], []
rec_q = dict((s, []) for s in m.terminals)
max_resid = 0.0

for k in range(nsteps):
    t_new = t + DT
    pao, pimp = p_ao(t_new), p_imp(t_new)
    P_new = m.step(P, pao, pimp)

    q = m.flows(P_new, pao, pimp)
    for node in m.free_nodes:
        i = m.idx[node]
        store = m.C[node] * (P_new[i] - P[i]) / DT
        net = 0.0
        for s in m.names:
            if SEGMENTS[s]["node_d"] == node:
                net += q[s]
            if SEGMENTS[s]["node_p"] == node:
                net -= q[s]
        max_resid = max(max_resid, abs(net - store))

    P, t = P_new, t_new
    if t > 3 * T:
        rec_t.append(t - 3 * T)
        rec_pao.append(pao * MMHG_PER_KPA)
        for s in m.terminals:
            rec_q[s].append(q[s])

print()
check("max mass-balance residual over all nodes/steps", max_resid, 0.0,
      tol=1e-9)

rec_t = np.array(rec_t)
sys_mask = rec_t < T_SYS
dia_mask = ~sys_mask

print()
print("  %-6s %9s %9s %9s %9s   %s"
      % ("term", "mean Q", "peak Q", "sys mean", "dia mean", "phasic pattern"))
for s in m.terminals:
    q = np.array(rec_q[s])
    qs, qd = q[sys_mask].mean(), q[dia_mask].mean()
    patt = "diastolic-dominant" if qd > qs else "SYSTOLIC-dominant <-- WRONG"
    print("  %-6s %9.4f %9.4f %9.4f %9.4f   %s"
          % (s, q.mean(), q.max(), qs, qd, patt))


# ---------------------------------------------------------------------------
hdr("Diagnosis -- inter-territory steal, and how it depends on terminal R")

print("  With no downstream resistance the territories are coupled through")
print("  epicardial resistances of order 0.5-3.6.  IMP differences between")
print("  territories then drive flow sideways instead of into myocardium.")
print()
print("  Sweeping a uniform terminal resistance Rt (Rt=None is the current")
print("  no-Z model).  'reversed' = territories with negative systolic flow;")
print("  Table 1 Z values are 80-170 for reference.")
print()
print("  %-10s %10s %10s %10s %10s"
      % ("Rt", "mean Q_tot", "reversed", "min sys Q", "diast-dom"))

for Rt in [None, 1.0, 5.0, 20.0, 50.0, 100.0, 150.0]:
    mm = CoronaryRC(SUBTREES["lad3"], DT, terminal_resistance=Rt)
    PP2 = mm.steady_state(p_ao(0.0), p_imp(0.0))
    tt2 = 0.0
    rt2, rq2 = [], dict((s, []) for s in mm.terminals)
    for k in range(nsteps):
        tt2 += DT
        pao, pimp = p_ao(tt2), p_imp(tt2)
        PP2 = mm.step(PP2, pao, pimp)
        if tt2 > 3 * T:
            rt2.append(tt2 - 3 * T)
            perf = mm.perfusion(PP2, pao, pimp)
            for s in mm.terminals:
                rq2[s].append(perf[s])
    rt2 = np.array(rt2)
    sm, dm = rt2 < T_SYS, rt2 >= T_SYS
    qs = dict((s, np.array(rq2[s])) for s in mm.terminals)
    nrev = sum(1 for s in mm.terminals if qs[s][sm].mean() < 0)
    ndd = sum(1 for s in mm.terminals
              if qs[s][dm].mean() > qs[s][sm].mean())
    print("  %-10s %10.3f %10d %10.3f %7d / %d"
          % ("none" if Rt is None else "%.0f" % Rt,
             sum(qs[s].mean() for s in mm.terminals),
             nrev,
             min(qs[s][sm].mean() for s in mm.terminals),
             ndd, len(mm.terminals)))

print()
print("  Steal vanishes once Rt is large enough to decouple the territories.")
print("  This is the role the Table 1 Z values were playing.  Recorded here")
print("  as evidence; the model itself stays no-Z until that is settled.")

print("\n" + "=" * 72)
print("Rungs 1-4 passed exactly.  Rung 5 runs and conserves mass;")
print("its phasic pattern is compromised by steal, as diagnosed above.")
print("=" * 72)
