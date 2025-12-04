import matplotlib.pyplot as plt
import sisl
import numpy as np
from typing import Union

import plotly.graph_objects as go
from warnings import warn

from .structure import guess_hexagon_center, find_nearest_atoms

# Force initialization of sisl plotting system
# Needed because utils.plots functions call .plot(), and
# Geometry.plot is lazy-loaded. This ensures plotting works
# even on first call after notebook kernel restart.
__dummyplot_initilizer__ = sisl.geom.graphene_nanoribbon(1).plot()


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
    size = kwargs.get("size", 50)
    Emax = kwargs.get("Emax", 2)
    Emin = kwargs.get("Emin", -2)
    if isinstance(E, (int, float)):
        return np.linspace(-E, E, size)
    elif E is None:
        return np.linspace(Emin, Emax, size)
    else:
        raise ValueError("E must be a number or None.")
    
def mark_electrode(lr_indices, scale_func=None):
    """Helper function for plotting electrodes.
    Colors 'left' and 'right' electrodes using blue and red respectively.
    Scales the atom sizes to match opposing electrodes.
    
    Parameters
    ---
    lr_indices : tuple or ndarray
        The indices to mark the electrodes
    scale_func : function
        How the electrode atom sizes scales by ribbon number index, defualt to `lambda x: 0.3*x + 0.3`
    
    Returns
    ---
    atoms_style : list[dict,]
        list of dictionaries for use in `sisl.Geometry.plot()`
    """
    if isinstance(lr_indices, tuple):
        if isinstance(lr_indices[0], np.ndarray) and isinstance(lr_indices[1], np.ndarray):
            left, right = lr_indices
            lr = np.stack([left, right])
    else:
        lr = np.asarray(lr_indices)
    
    # Case: squeezed shape (2, N) for a single ribbon
    if lr.ndim == 2 and lr.shape[0] == 2:
        lr = lr[:, np.newaxis, :] # add a K-dimension for the number of ribbons -> (2, 1, N)
    
    if lr.ndim != 3 or lr.shape[0] != 2:
        raise ValueError(f"Invalid lr_indices shape {lr.shape}, "
                         "expected (2, K, N) or a tuple of two arrays")
        
    if scale_func is None:
        scale_func = lambda x: 0.3*(1 + x)
    _, num_ribbons, _ = lr.shape
    
    atoms_style = []
    for i in range(num_ribbons):
        l = lr[0, i]
        r = lr[1, i]
        atoms_style += [{"atoms": l, "size": scale_func(i), "color": "blue"}]
        atoms_style += [{"atoms": r, "size": scale_func(i), "color": "red"}]
    return atoms_style
    
def plot_with_center(structure: sisl.Geometry, **kwargs):
    """Plot structure and highlight central hexagon + geometric center. 
    
    Note
    ---
    Only intended for plotting nanoribbon. For reduced graphene structure, the *plotted* rotation center is not guaranteed correct.
    """
    atoms_style = kwargs.get("atoms_style", [])
    axes = kwargs.get("axes", "xy")
    bind_bonds_to_ats = kwargs.get("bind_bonds_to_ats", True)
    
   
    if isinstance(structure, sisl.Geometry):
        # print("input type is a Geometry")
        fig = structure.plot(axes=axes, bind_bonds_to_ats=bind_bonds_to_ats)
    else:
        print("input type is a figure!")
        fig = structure
        atoms_style = [structure.inputs["atoms_style"]]
        structure = structure.inputs["geometry"]
        warn("\nThis is not implemented yet")
        raise NotImplementedError("This does not function with plotly's scatterplot!!!! needs fix")
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