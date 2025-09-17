__version__ = "0.1.0"

import sisl
import numpy as np
from ._wrappers import timeit
from .structure import generate_structure

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

def hamiltonian(structure, **kwargs):
    """Create a tight-binding Hamiltonian for a given structure."""
    r = kwargs.get("r", (0, 1.44)) # bond length in Angstrom
    t = kwargs.get("t", (0.0, -2.7)) # energy in eV
    H = sisl.Hamiltonian(structure)
    H.construct([r, t])
    if kwargs.get("finalize", False):
        H.finalize() # make hamiltonian sparse
    return H

@timeit
def compute_dos(width=5, length=5, **kwargs):
    E = kwargs.get("E", None)
    if E is None:
        E = _parse_E_range(**kwargs)
    k = kwargs.get("k", (0,0,0))
    graphene, _ = generate_structure(width=width, length=length)

    H = hamiltonian(graphene, **kwargs)
    es = H.eigenstate(k=k)
    return es.DOS(E)

