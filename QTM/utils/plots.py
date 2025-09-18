import matplotlib.pyplot as plt
import sisl

import plotly.graph_objects as go
from .structure import guess_hexagon_center
from .energies import _parse_E_range, hamiltonian, compute_dos
# ---------------------------
# Plotting utilities
# ---------------------------

def plot_with_center(structure, **KWARGS):
    """Plot structure and highlight central hexagon + geometric center."""
    
    KWARGS.setdefault("axes", "xy")
    KWARGS.setdefault("bind_bonds_to_ats", True)
    atom_idx, hex_center = guess_hexagon_center(structure)
    # print(f"Hexagon atom indices: {atom_idx}")
    geom_center = structure.center()

    fig = structure.plot(**KWARGS)

    # highlight hexagon atoms
    fig.update_inputs(atoms_style={"color": "red", "atoms": atom_idx.tolist()})

    # add markers
    fig.add_trace(go.Scatter(
        x=[geom_center[0]], y=[geom_center[1]],
        mode="markers", marker=dict(size=10, color="red", symbol="x"),
        name="Geometric Center"
    ))
    
    fig.add_trace(go.Scatter(
        x=[hex_center[0]], y=[hex_center[1]],
        mode="markers", marker=dict(size=10, color="blue", symbol="circle"),
        name="Rotation Center"
    ))

    return fig

def plot_DOS(width=5, length=5, **kwargs):
    E = _parse_E_range(**kwargs)
    kwargs["E"] = E
    dos = compute_dos(width, length, **kwargs)
    # print("DOS computed! Plotting...", end="\r")
    ## plot figure
    plt.figure(figsize=(6,4))
    plt.plot(E, dos, color="black")
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.title(f"Graphene Nanoribbon DOS (width={width}, length={length})")
    plt.grid(True)
    plt.show()
    
def plot_center_pdos(structure, radius, **kwargs):
    if (radius < 2):
        print(f"Radius too small ({radius}), using radius=2")
        radius = 2
    center = structure.center() # xyz coordinate for center
    center_atoms = structure.close(center, radius) # C atom index within radius of center 
    E = _parse_E_range(**kwargs) # make list for energy values
    k = kwargs.get("k", (1,1,1)) # get k point to evaluate at, default: only Gamma (0,0,0)
    
    orb_groups = [
        {"atoms" : center_atoms}
    ] # group of atoms to use for plot
    
    # make hamiltonian
    H = hamiltonian(structure, **kwargs)
    
    plot_range = kwargs.get("range", (E[0], E[-1]))
    size = kwargs.get("size", len(E))
    pdos_plot = H.plot.pdos(kgrid=k, data_Erange=(E[0], E[-1]), Erange=plot_range, nE=size)
    
    pdos_plot.update_inputs(groups=orb_groups)
    fig_title = f"PDOS,  r={radius:.1f} [Å]"
    pdos_plot.update_layout(title=dict(text=fig_title, font=dict(size=30)))
    pdos_plot.show()