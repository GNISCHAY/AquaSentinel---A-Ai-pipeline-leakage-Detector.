"""
Plotly figures for the dashboard (Member 3 - Frontend & Dashboard Developer).

Every function takes plain data and returns a plotly Figure, so charts can be
built and tested without running Streamlit.
"""

import numpy as np
import plotly.graph_objects as go

from aquasentinel.network import RESERVOIR

# Water-technology colour palette
NAVY = "#0B1F33"
PANEL = "#10283F"
GRID = "#1E3A56"
TEXT = "#D6E6F2"
MUTED = "#7F9BB3"
WATER = "#2E86C1"
TEAL = "#00B4D8"
GREEN = "#06D6A0"
AMBER = "#FFB703"
RED = "#EF476F"
WHITE = "#FFFFFF"


def _base_layout(fig, title=None, height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 20, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def network_figure(network, pipe_importance, latest_pressures=None, predicted_pipe=None,
                   suspicious_pipes=(), pipe_scores=None, true_leak_pipe=None, height=520):
    """
    Digital twin map.
      * pipe colour : blue = normal, amber = suspicious, red = predicted leak
      * pipe width  : diameter
      * junction colour : latest pressure (if given)
      * ring around a junction = pressure sensor, diamond on a pipe = flow meter
      * white X = true leak location (only when the presenter reveals it)
    """
    pipe_scores = pipe_scores or {}
    fig = go.Figure()
    pos = network.positions

    # --- Pipes (one trace per pipe so each can have its own colour/width) ---
    for k, pipe in enumerate(network.pipe_names):
        start, end = network.pipe_start[k], network.pipe_end[k]
        (x0, y0), (x1, y1) = pos[start], pos[end]
        if pipe == predicted_pipe:
            colour, width = RED, 9
        elif pipe in suspicious_pipes:
            colour, width = AMBER, 6
        else:
            colour, width = WATER, 2 + network.diameters_m[k] * 25
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=colour, width=width),
                                 hoverinfo="skip", showlegend=False))

    # Invisible markers at pipe mid-points carry the hover text
    mid_x, mid_y, hover = [], [], []
    for k, pipe in enumerate(network.pipe_names):
        (x0, y0), (x1, y1) = pos[network.pipe_start[k]], pos[network.pipe_end[k]]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        score = pipe_scores.get(pipe)
        hover.append(
            f"<b>{pipe}</b> ({network.pipe_start[k]} - {network.pipe_end[k]})<br>"
            f"Diameter: {network.diameters_m[k] * 1000:.0f} mm<br>"
            f"Length: {network.lengths_m[k]:.0f} m<br>"
            f"Importance: {pipe_importance.get(pipe, 0):.2f}"
            + (f"<br>AI leak score: {score:.2f}" if score is not None else "")
        )
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="text", text=network.pipe_names,
                             textfont=dict(size=10, color=MUTED), hovertext=hover,
                             hoverinfo="text", showlegend=False))

    # --- Flow meters --------------------------------------------------------
    meter_x, meter_y = [], []
    for pipe in network.flow_meter_pipes:
        k = network.pipe_index(pipe)
        (x0, y0), (x1, y1) = pos[network.pipe_start[k]], pos[network.pipe_end[k]]
        meter_x.append(x0 + 0.3 * (x1 - x0))
        meter_y.append(y0 + 0.3 * (y1 - y0))
    fig.add_trace(go.Scatter(x=meter_x, y=meter_y, mode="markers", name="Flow meter",
                             marker=dict(symbol="diamond", size=13, color=TEAL,
                                         line=dict(color=WHITE, width=1)),
                             hovertext=[f"Flow meter on {p}" for p in network.flow_meter_pipes],
                             hoverinfo="text"))

    # --- Junctions ------------------------------------------------------------
    names = network.junction_names
    xs = [pos[n][0] for n in names]
    ys = [pos[n][1] for n in names]
    sensor_ring = [3 if n in network.pressure_sensor_nodes else 0 for n in names]
    hover = []
    for i, n in enumerate(names):
        text = f"<b>{n}</b><br>Elevation: {network.elevations_m[i]:.0f} m<br>Avg demand: {network.base_demands_lps[i]:.1f} L/s"
        if latest_pressures is not None:
            text += f"<br>Pressure (sim.): {latest_pressures[i]:.1f} m"
        if n in network.pressure_sensor_nodes:
            text += "<br>Pressure sensor installed"
        if n in network.critical_facilities:
            text += f"<br>{network.critical_facilities[n]}"
        hover.append(text)
    marker = dict(size=17, line=dict(color=TEAL, width=sensor_ring), color=MUTED)
    if latest_pressures is not None:
        marker.update(color=latest_pressures, colorscale="Blues", reversescale=True, cmin=15, cmax=65,
                      colorbar=dict(title="Pressure (m)", thickness=12, len=0.6, x=1.0))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=names,
                             textposition="top left", textfont=dict(size=11, color=TEXT),
                             marker=marker, hovertext=hover, hoverinfo="text",
                             name="Junction (ring = pressure sensor)"))

    # Critical facilities
    crit = list(network.critical_facilities)
    fig.add_trace(go.Scatter(x=[pos[n][0] + 0.18 for n in crit], y=[pos[n][1] + 0.18 for n in crit],
                             mode="markers", name="Critical facility",
                             marker=dict(symbol="star", size=13, color=GREEN),
                             hovertext=[network.critical_facilities[n] for n in crit], hoverinfo="text"))

    # Reservoir
    rx, ry = pos[RESERVOIR]
    fig.add_trace(go.Scatter(x=[rx], y=[ry], mode="markers+text", text=["R1"],
                             textposition="bottom center", name="Reservoir",
                             marker=dict(symbol="square", size=30, color=TEAL,
                                         line=dict(color=WHITE, width=2)),
                             hovertext=["Reservoir R1 - supply head 75 m"], hoverinfo="text"))

    # True leak (ground truth) - only when revealed
    if true_leak_pipe:
        k = network.pipe_index(true_leak_pipe)
        (x0, y0), (x1, y1) = pos[network.pipe_start[k]], pos[network.pipe_end[k]]
        fig.add_trace(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2 + 0.14], mode="markers",
                                 name="True leak (hidden from AI)",
                                 marker=dict(symbol="x", size=16, color=WHITE,
                                             line=dict(color=RED, width=2))))

    # Legend entries for pipe colours
    for label, colour in (("Normal pipe", WATER), ("Suspicious pipe", AMBER), ("Predicted leak", RED)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label,
                                 line=dict(color=colour, width=5)))

    _base_layout(fig, height=height)
    fig.update_layout(plot_bgcolor=NAVY, hovermode="closest",
                      legend=dict(orientation="h", y=-0.02, yanchor="top", font=dict(size=11)),
                      margin=dict(l=10, r=10, t=10, b=70))
    fig.update_xaxes(visible=False, range=[-1.5, 3.5])
    fig.update_yaxes(visible=False, range=[-0.4, 3.45])
    return fig


def anomaly_timeline(time_hours, scores, threshold, alarm, leak_start_hour=None, height=300):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_hours, y=scores, mode="lines", name="Anomaly score",
                             line=dict(color=TEAL, width=2), fill="tozeroy",
                             fillcolor="rgba(0,180,216,0.12)"))
    alarm_hours = np.asarray(time_hours)[np.asarray(alarm)]
    alarm_scores = np.asarray(scores)[np.asarray(alarm)]
    fig.add_trace(go.Scatter(x=alarm_hours, y=alarm_scores, mode="markers", name="Alarm",
                             marker=dict(color=RED, size=6)))
    fig.add_hline(y=threshold, line=dict(color=AMBER, dash="dash"),
                  annotation_text="alarm threshold", annotation_font_color=AMBER)
    if leak_start_hour is not None:
        fig.add_vline(x=leak_start_hour, line=dict(color=WHITE, dash="dot"),
                      annotation_text="true leak start", annotation_font_color=WHITE)
    _base_layout(fig, "Anomaly score – Isolation Forest + range check (1.0 = threshold)", height)
    fig.update_xaxes(title="Hour of day")
    return fig


def sensor_chart(time_hours, measured, expected, column, leak_start_hour=None,
                 repaired=None, height=280):
    unit = "m" if column.startswith("pressure_") else "L/s"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_hours, y=expected[column], name="Digital twin (expected)",
                             line=dict(color=MUTED, dash="dash")))
    if repaired is not None:
        fig.add_trace(go.Scatter(x=time_hours, y=repaired[column], name="After repair (simulated)",
                                 line=dict(color=GREEN, width=1.5)))
    fig.add_trace(go.Scatter(x=time_hours, y=measured[column], name="Measured (simulated)",
                             line=dict(color=TEAL, width=2), connectgaps=False))
    if leak_start_hour is not None:
        fig.add_vline(x=leak_start_hour, line=dict(color=WHITE, dash="dot"))
    _base_layout(fig, column.replace("_", " ").title(), height)
    fig.update_yaxes(title=unit)
    fig.update_xaxes(title="Hour of day")
    return fig


def candidates_bar(ranking, top_n=6, height=300):
    top = ranking.head(top_n).iloc[::-1]
    colours = [RED if i == len(top) - 1 else AMBER for i in range(len(top))]
    fig = go.Figure(go.Bar(x=top["score"], y=top["pipe"], orientation="h",
                           marker_color=colours, text=[f"{s:.2f}" for s in top["score"]],
                           textposition="outside"))
    _base_layout(fig, "Most likely leaking pipes (Random Forest vote share)", height)
    fig.update_xaxes(range=[0, 1.1], title="Score (not a calibrated probability)")
    return fig


def contribution_bar(explanation, height=300):
    top = explanation.iloc[::-1]
    fig = go.Figure(go.Bar(x=top["contribution_share"], y=top["sensor"], orientation="h",
                           marker_color=TEAL,
                           hovertext=[f"deviation {z:+.1f} sigma" for z in top["deviation_z"]],
                           text=[f"{z:+.1f} sigma" for z in top["deviation_z"]],
                           textposition="outside"))
    _base_layout(fig, "Which sensors drove the prediction", height)
    fig.update_xaxes(title="Share of explanation", range=[0, max(top["contribution_share"].max() * 1.3, 0.1)])
    return fig


def confusion_heatmap(labels, matrix, height=560):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels,
                               colorscale=[[0, PANEL], [0.25, WATER], [1, TEAL]],
                               hovertemplate="True %{y}<br>Predicted %{x}<br>Days %{z}<extra></extra>"))
    _base_layout(fig, "Leak localisation confusion matrix (baseline test days)", height)
    fig.update_xaxes(title="Predicted pipe", tickangle=-45)
    fig.update_yaxes(title="True pipe", autorange="reversed")
    return fig


def importance_bar(importance_by_sensor, height=380):
    items = sorted(importance_by_sensor.items(), key=lambda item: item[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=TEAL))
    _base_layout(fig, "Random Forest global feature importance", height)
    fig.update_xaxes(title="Mean decrease in impurity")
    return fig


def comparison_bar(categories, values_a, values_b, label_a, label_b, title, height=320):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values_a, name=label_a, marker_color=TEAL))
    fig.add_trace(go.Bar(x=categories, y=values_b, name=label_b, marker_color=AMBER))
    _base_layout(fig, title, height)
    fig.update_layout(barmode="group")
    return fig
