"""
Water network construction (Member 2 - Simulation Developer).

Builds a small SYNTHETIC water distribution network:
  * 1 reservoir (R1) that supplies the whole network
  * 16 junctions (J1..J16) laid out on a 4 x 4 grid, each with customer demand
  * 22 pipes (P1..P22): P1 is the trunk main from the reservoir
  * 6 pressure sensors and 3 flow meters (deliberately LIMITED coverage)

The network is stored twice:
  * as a NetworkX graph (for graph algorithms and drawing)
  * as NumPy arrays (for the fast hydraulic solver)
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

RESERVOIR = "R1"

# Junction coordinates on a 4 x 4 grid (x = column, y = row; 1 unit = 400 m)
JUNCTION_POSITIONS = {
    f"J{row * 4 + col + 1}": (float(col), float(3 - row))
    for row in range(4)
    for col in range(4)
}

# Ground elevation of every junction in metres (gently rising to the south-east)
JUNCTION_ELEVATIONS = {
    "J1": 10, "J2": 12, "J3": 14, "J4": 17,
    "J5": 11, "J6": 13, "J7": 16, "J8": 19,
    "J9": 12, "J10": 15, "J11": 18, "J12": 21,
    "J13": 14, "J14": 17, "J15": 20, "J16": 23,
}

# Average customer demand of every junction in litres per second (L/s)
JUNCTION_BASE_DEMANDS = {
    "J1": 0.6, "J2": 0.9, "J3": 0.8, "J4": 0.7,
    "J5": 1.0, "J6": 1.4, "J7": 1.1, "J8": 0.8,
    "J9": 0.9, "J10": 1.2, "J11": 1.0, "J12": 0.7,
    "J13": 0.8, "J14": 1.1, "J15": 1.3, "J16": 0.9,
}

# (pipe name, start node, end node, diameter in metres)
PIPE_DEFINITIONS = [
    ("P1", "R1", "J1", 0.25),    # trunk main from the reservoir
    # horizontal pipes
    ("P2", "J1", "J2", 0.15), ("P3", "J2", "J3", 0.10), ("P4", "J3", "J4", 0.08),
    ("P5", "J5", "J6", 0.10), ("P6", "J6", "J7", 0.10), ("P7", "J7", "J8", 0.08),
    ("P8", "J9", "J10", 0.08), ("P9", "J11", "J12", 0.08),
    ("P10", "J13", "J14", 0.08), ("P11", "J15", "J16", 0.08),
    # vertical pipes
    ("P12", "J1", "J5", 0.15), ("P13", "J5", "J9", 0.10), ("P14", "J9", "J13", 0.08),
    ("P15", "J2", "J6", 0.10), ("P16", "J6", "J10", 0.10), ("P17", "J10", "J14", 0.08),
    ("P18", "J3", "J7", 0.08), ("P19", "J4", "J8", 0.08), ("P20", "J8", "J12", 0.08),
    ("P21", "J12", "J16", 0.08), ("P22", "J11", "J15", 0.08),
]

# Pipes that are not on the grid spacing get a custom length (metres)
PIPE_LENGTH_OVERRIDES = {"P1": 600.0}
GRID_SPACING_M = 400.0

# Sensor placement: few sensors, as in a real utility with limited budget
PRESSURE_SENSOR_NODES = ["J4", "J6", "J9", "J11", "J13", "J16"]
FLOW_METER_PIPES = ["P1", "P5", "P20"]

# Junctions that serve critical facilities (all fictional)
CRITICAL_FACILITIES = {"J6": "Hospital (simulated)", "J15": "School (simulated)"}


@dataclass
class WaterNetwork:
    """Everything the rest of the project needs to know about the network."""

    graph: nx.Graph
    junction_names: list
    pipe_names: list
    pipe_start: list
    pipe_end: list
    lengths_m: np.ndarray
    diameters_m: np.ndarray
    elevations_m: np.ndarray
    base_demands_lps: np.ndarray
    positions: dict
    pressure_sensor_nodes: list = field(default_factory=list)
    flow_meter_pipes: list = field(default_factory=list)
    critical_facilities: dict = field(default_factory=dict)

    # Incidence matrices (filled in by build_network)
    incidence_junctions: np.ndarray = None   # shape (num_pipes, num_junctions)
    incidence_reservoir: np.ndarray = None   # shape (num_pipes,)

    @property
    def num_pipes(self):
        return len(self.pipe_names)

    @property
    def num_junctions(self):
        return len(self.junction_names)

    def junction_index(self, name):
        return self.junction_names.index(name)

    def pipe_index(self, name):
        return self.pipe_names.index(name)

    def sensor_feature_names(self):
        """Names of every sensor channel, in a fixed order used everywhere."""
        pressure = [f"pressure_{node}" for node in self.pressure_sensor_nodes]
        flow = [f"flow_{pipe}" for pipe in self.flow_meter_pipes]
        return pressure + flow

    def neighbouring_pipes(self, pipe_name):
        """Pipes that share a junction with the given pipe (1 hop away)."""
        start = self.pipe_start[self.pipe_index(pipe_name)]
        end = self.pipe_end[self.pipe_index(pipe_name)]
        neighbours = set()
        for other, s, e in zip(self.pipe_names, self.pipe_start, self.pipe_end):
            if other != pipe_name and ({s, e} & {start, end}) - {RESERVOIR}:
                neighbours.add(other)
        return neighbours


def build_network():
    """Create the synthetic network and return a WaterNetwork object."""
    junction_names = list(JUNCTION_POSITIONS.keys())
    positions = dict(JUNCTION_POSITIONS)
    positions[RESERVOIR] = (-1.0, 3.0)

    graph = nx.Graph()
    graph.add_node(RESERVOIR, kind="reservoir", pos=positions[RESERVOIR])
    for name in junction_names:
        graph.add_node(
            name,
            kind="junction",
            pos=positions[name],
            elevation=JUNCTION_ELEVATIONS[name],
            base_demand=JUNCTION_BASE_DEMANDS[name],
            has_pressure_sensor=name in PRESSURE_SENSOR_NODES,
            facility=CRITICAL_FACILITIES.get(name, ""),
        )

    pipe_names, pipe_start, pipe_end, lengths, diameters = [], [], [], [], []
    for name, start, end, diameter in PIPE_DEFINITIONS:
        length = PIPE_LENGTH_OVERRIDES.get(name, GRID_SPACING_M)
        graph.add_edge(
            start, end, name=name, length=length, diameter=diameter,
            has_flow_meter=name in FLOW_METER_PIPES,
        )
        pipe_names.append(name)
        pipe_start.append(start)
        pipe_end.append(end)
        lengths.append(length)
        diameters.append(diameter)

    network = WaterNetwork(
        graph=graph,
        junction_names=junction_names,
        pipe_names=pipe_names,
        pipe_start=pipe_start,
        pipe_end=pipe_end,
        lengths_m=np.array(lengths),
        diameters_m=np.array(diameters),
        elevations_m=np.array([JUNCTION_ELEVATIONS[j] for j in junction_names], dtype=float),
        base_demands_lps=np.array([JUNCTION_BASE_DEMANDS[j] for j in junction_names]),
        positions=positions,
        pressure_sensor_nodes=list(PRESSURE_SENSOR_NODES),
        flow_meter_pipes=list(FLOW_METER_PIPES),
        critical_facilities=dict(CRITICAL_FACILITIES),
    )

    # Incidence matrix: +1 where a pipe starts, -1 where it ends.
    # A positive flow therefore means "flowing from start to end".
    incidence = np.zeros((network.num_pipes, network.num_junctions))
    incidence_reservoir = np.zeros(network.num_pipes)
    for k, (start, end) in enumerate(zip(pipe_start, pipe_end)):
        for node, sign in ((start, 1.0), (end, -1.0)):
            if node == RESERVOIR:
                incidence_reservoir[k] = sign
            else:
                incidence[k, network.junction_index(node)] = sign
    network.incidence_junctions = incidence
    network.incidence_reservoir = incidence_reservoir
    return network


def compute_pipe_importance(network):
    """
    Score how important each pipe is to the network, from 0 (low) to 1 (high).

    Importance = 0.4 x average flow carried     (how much water it moves)
               + 0.3 x edge betweenness         (how many routes rely on it)
               + 0.3 x critical facility served (hospital / school nearby)

    The weights are design choices; each part is scaled to 0..1 first.
    Returns a dict {pipe_name: importance}.
    """
    from aquasentinel.hydraulics import solve_hydraulics  # local import avoids a cycle

    demands = network.base_demands_lps[np.newaxis, :]
    result = solve_hydraulics(network, demands)
    average_flow = np.abs(result["flows_lps"][0])
    flow_score = average_flow / average_flow.max()

    betweenness = nx.edge_betweenness_centrality(network.graph, normalized=True)
    betweenness_by_pipe = np.array([
        betweenness.get((s, e), betweenness.get((e, s), 0.0))
        for s, e in zip(network.pipe_start, network.pipe_end)
    ])
    betweenness_score = betweenness_by_pipe / betweenness_by_pipe.max()

    critical_score = np.array([
        1.0 if (s in network.critical_facilities or e in network.critical_facilities) else 0.0
        for s, e in zip(network.pipe_start, network.pipe_end)
    ])

    importance = 0.4 * flow_score + 0.3 * betweenness_score + 0.3 * critical_score
    return {name: round(float(v), 3) for name, v in zip(network.pipe_names, importance)}
