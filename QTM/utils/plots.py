import matplotlib.pyplot as plt
import sisl
import numpy as np

import plotly.graph_objects as go
from warnings import warn

from .structure import guess_hexagon_center, find_nearest_atoms
# ---------------------------
# Plotting utilities
# ---------------------------
def _parse_E_range(**kwargs):
    """
    Parsing energy value range from keywords
    If E is a single number, create a symmetric range around zero with that value as the maximum absolute value.
    If E is None, use default Emin and Emax values.
    """
    E = kwargs.get("E", None)
    size = kwargs.get("size", 400)
    Emax = kwargs.get("Emax", 4)
    Emin = kwargs.get("Emin", -4)
    if isinstance(E, (int, float)):
        return np.linspace(-E, E, size)
    elif E is None:
        return np.linspace(Emin, Emax, size)
    else:
        raise ValueError("E must be a number or None.")
    
def mark_electrode(lr_indices):
    """Helper function for plotting electrodes.
    Colors 'left' and 'right' electrodes using blue and red respectively.
    Scales the atom sizes to match opposing electrodes."""
    atoms_style = []
    for i, (l, r) in enumerate(zip(*lr_indices)):
        atoms_style += [{"atoms": l, "size": i*0.3 + 0.3, "color": "blue"}]
        atoms_style += [{"atoms": r, "size": i*0.3 + 0.3, "color": "red"}]
    return atoms_style
    
def plot_with_center(structure: sisl.Geometry | sisl.viz.plots.GeometryPlot, **KWARGS):
    """Plot structure and highlight central hexagon + geometric center. 
    
    Note
    ---
    Only intended for plotting nanoribbon. For reduced graphene structure, the *plotted* rotation center is not guaranteed correct.
    """
    atoms_style = KWARGS.get("atoms_style", [])
    axes = KWARGS.get("axes", "xy")
    bind_bonds_to_ats = KWARGS.get("bind_bonds_to_ats", True)
    
    if isinstance(structure, sisl.viz.plots.GeometryPlot):
        print("input type is a figure!")
        fig = structure
        atoms_style = [structure.inputs["atoms_style"]]
        structure = structure.inputs["geometry"]
        warn("\nThis is not implemented yet")
        raise NotImplementedError("This does not function with plotly's scatterplot!!!! needs fix")
    elif isinstance(structure, sisl.Geometry):
        # print("input type is a Geometry")
        fig = structure.plot(axes=axes, bind_bonds_to_ats=bind_bonds_to_ats)
    
    # print(f"{type(structure) = }")
    # print(f"{type(fig) = }")
    hex_center = guess_hexagon_center(structure)
    atom_idx = find_nearest_atoms(structure.xyz, hex_center, 6)
    geom_center = structure.center()

    
    # highlight hexagon atoms
    atoms_style += [{"color": "violet", "atoms": atom_idx.tolist()}]
    
    # color atoms
    fig.update_inputs(atoms_style=atoms_style)

    # add markers
    fig.add_trace(go.Scatter(
        x=[geom_center[0]], y=[geom_center[1]],
        mode="markers", marker=dict(size=12, color="red", symbol="x"),
        name="Geometric Center"
    ))
    
    fig.add_trace(go.Scatter(
        x=[hex_center[0]], y=[hex_center[1]],
        mode="markers", marker=dict(size=9, color="violet", symbol="diamond"),
        name="Rotation Center"
    ))

    return fig 