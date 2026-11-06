"""
Simplified steady-state hydraulic solver (Member 2 - Simulation Developer).

HONEST DISCLAIMER
-----------------
This is a teaching-grade solver, NOT a validated hydraulic model such as
EPANET. It uses two real engineering ideas:

1. Hazen-Williams head loss:  h_loss = r * Q * |Q|^0.852
   with r = 10.67 * L / (C^1.852 * D^4.87)   (SI units, Q in m^3/s)

2. The "linear theory method" (Wood & Charles, 1972): the non-linear pipe
   equation is linearised around the current flow guess, the resulting
   linear system (like an electrical circuit) is solved, and the process
   repeats until flows stop changing.

Leaks are modelled as a pressure-dependent outflow at the middle of a pipe:
    leak_flow (L/s) = leak_coefficient * sqrt(pressure in metres)
The leak flow is shared equally between the two junctions at the pipe ends.

Each call solves many time steps at once (vectorised with NumPy), which
keeps data generation fast enough for a laptop.
"""

import numpy as np

from aquasentinel import config

MIN_FLOW_M3S = 1e-7          # avoids division by zero for (almost) stagnant pipes
DAMPING = 0.5                # average old and new guesses -> stable convergence


def _resistance(network):
    """Hazen-Williams resistance coefficient r for every pipe."""
    c = config.HAZEN_WILLIAMS_C
    return 10.67 * network.lengths_m / (c ** 1.852 * network.diameters_m ** 4.87)


def _leak_to_junction_matrix(network):
    """
    Matrix (num_junctions x num_pipes) that sends each pipe's leak flow to its
    end junctions: 50/50, or 100% to the junction if the other end is the reservoir.
    """
    share = np.zeros((network.num_junctions, network.num_pipes))
    for k in range(network.num_pipes):
        ends = [j for j in np.flatnonzero(network.incidence_junctions[k])]
        for j in ends:
            share[j, k] = 1.0 / len(ends)
    return share


def solve_hydraulics(network, demands_lps, leak_coefficients=None):
    """
    Solve the network for T time steps at once.

    Parameters
    ----------
    network : WaterNetwork
    demands_lps : array (T, num_junctions) - customer demand at each junction
    leak_coefficients : array (T, num_pipes) or None - leak coefficient of
        every pipe at every time step (0 = no leak)

    Returns
    -------
    dict with arrays:
        pressures_m     (T, num_junctions)
        flows_lps       (T, num_pipes)   positive = start -> end
        leak_flows_lps  (T, num_pipes)
        inflow_lps      (T,)             water leaving the reservoir
    """
    demands_lps = np.atleast_2d(np.asarray(demands_lps, dtype=float))
    num_steps = demands_lps.shape[0]
    if leak_coefficients is None:
        leak_coefficients = np.zeros((num_steps, network.num_pipes))

    a_j = network.incidence_junctions          # (pipes, junctions)
    a_r = network.incidence_reservoir          # (pipes,)
    head_res = config.RESERVOIR_HEAD_M
    resistance = _resistance(network)          # (pipes,)
    leak_share = _leak_to_junction_matrix(network)

    demands_m3s = demands_lps / 1000.0
    flows = np.full((num_steps, network.num_pipes), 1e-3)   # initial guess
    leak_m3s = np.zeros((num_steps, network.num_pipes))
    pressures = np.zeros((num_steps, network.num_junctions))

    for _ in range(config.SOLVER_ITERATIONS):
        # 1) Linearise every pipe: Q = g * (head difference)
        conductance = 1.0 / (resistance * np.maximum(np.abs(flows), MIN_FLOW_M3S) ** 0.852)

        # 2) Build and solve the linear "circuit" for the junction heads
        total_outflow = demands_m3s + leak_m3s @ leak_share.T
        system = np.einsum("ki,tk,kj->tij", a_j, conductance, a_j)
        rhs = -total_outflow - np.einsum("ki,tk->ti", a_j, conductance * a_r * head_res)
        heads = np.linalg.solve(system, rhs[..., np.newaxis])[..., 0]

        # 3) New pipe flows from the head differences, with damping
        new_flows = conductance * (heads @ a_j.T + a_r * head_res)
        flows = DAMPING * flows + (1 - DAMPING) * new_flows

        # 4) Update leak outflow from the pressure at the middle of each pipe
        pressures = heads - network.elevations_m
        pipe_mid_pressure = pressures @ leak_share   # average of end pressures
        new_leak = leak_coefficients * np.sqrt(np.maximum(pipe_mid_pressure, 0.0)) / 1000.0
        leak_m3s = DAMPING * leak_m3s + (1 - DAMPING) * new_leak

    flows_lps = flows * 1000.0
    inflow_lps = (flows_lps * a_r).sum(axis=1)   # +1 where pipe starts at reservoir
    return {
        "pressures_m": pressures,
        "flows_lps": flows_lps,
        "leak_flows_lps": leak_m3s * 1000.0,
        "inflow_lps": inflow_lps,
    }
