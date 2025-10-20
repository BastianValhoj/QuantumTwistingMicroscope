import numpy as np
import sisl
from ._wrappers import timeit
from numba import njit
from tqdm import tqdm

from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
from sisl.typing import GeometryLike

from sisl.physics import RecursiveSI

# ====================================
# Helper function for parsing k-points
# ====================================
def _direction(Nk: int = 1, axis: int = 1) -> list[int]:
    """Infer direction for generating k-point mesh

    Parameters
    ----------
    Nk : int, optional
        Number of k-points (must be >= 1), by default 1
    axis : int, optional
        Axis index (0, 1, or 2) to extend, by default 1

    Returns
    -------
    d : list[int]
        The direction for k-point sampling

    Raises
    ------
    ValueError
        If `axis` is not one of 0, 1, 2 or `Nk` < 1.
    """
    if axis not in (0, 1, 2):
        raise ValueError("'axis' must be 0, 1, or 2.")
    if not isinstance(Nk, int) or Nk < 1:
        raise ValueError("'Nk' must be an integer >= 1.")
    d = [1, 1, 1]
    d[axis] = Nk
    return d

def hamiltonian(structure: sisl.Geometry, 
                *, r=(0, 1.44), t=(0.0, -2.7), finalize: bool = False) -> sisl.Hamiltonian:
    """Create a tight-binding Hamiltonian for a given structure.

    Parameters
    ----------
    structure : Geometry
        The device to find the Hamiltonian
    r : tuple, optional
        Bond paramter input for sisl.construct (kept as (0, bond_length) by default)
    t : tuple, optional
        Hopping/energy parameter for sisl.construct (kept as (0.0, -2.7) by default)
    finalize : bool, optional
        If True call `H.finalize()` before (makes sparse/optimized representation)

    Returns
    -------
    Hamiltonian
        The constructed Hamiltonian
    """
    H = sisl.Hamiltonian(structure)
    H.construct([r, t])
    if finalize:
        H.finalize() # make hamiltonian sparse
    return H


def lr_energies(electrode: sisl.Geometry | sisl.Hamiltonian | RecursiveSI, 
                En: complex = 1e-5j, kvec: Sequence = [0,0,0]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the left/right self-energy.

    Parameters
    ----------
    electrode : Geometry  |  Hamiltonian  |  RecursiveSI
        The electrode to find self-energies.
    En : complex, optional
        The energy for the self-energy calculation, by default 1e-5j
    kvec : list, optional
        The kpoint for the self-energy, by default [0,0,0]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left and right self-energies

    Raises
    ------
    TypeError
        If electrode is of invalid type
    """
    if not isinstance(kvec, list):
        kvec = list(kvec)
    if isinstance(electrode, sisl.Geometry):
        H_0 = hamiltonian(electrode)
        SE = RecursiveSI(H_0, infinite="+A")
    elif isinstance(electrode, sisl.physics.RecursiveSI):
        SE = electrode
    elif isinstance(electrode, sisl.Hamiltonian):
        SE = RecursiveSI(electrode, infinite="+A")
    else:
        raise TypeError("argument 'electrode' must be Geometry, Hamiltonian, or RecursiveSI")
    SE_L, SE_R = SE.self_energy_lr(E=En, k=kvec)
    return SE_L, SE_R

def add_lr_energies(Hk: np.ndarray,
                    lr_energies: tuple[np.ndarray, np.ndarray],
                    electrode_indices: tuple[np.ndarray, np.ndarray]
                    ) -> np.ndarray:
    """Add left and right self-energies to the input hamiltonian

    Parameters
    ----------
    Hk : np.ndarray
        The input hamiltonian of shape (N, N)
    lr_energies : tuple[ndarray, ndarray]
        Tuple of left and right self-energy arrays of shape (K, K)
    electrode_indices : tuple[ndarray, ndarray]
        Tuple of left and right electrode atom indices

    Returns
    -------
    ndarray
        The updated Hamiltonian
    """
    SE_L, SE_R = lr_energies
    left_indices, right_indices = electrode_indices
    for lidx, ridx in zip(left_indices, right_indices):
        Hk[np.ix_(lidx, lidx)] += SE_L
        Hk[np.ix_(ridx, ridx)] += SE_R
    return Hk

@njit
def _inverse(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(mat)

def compute_greens(Hk_with_lr_energies: np.ndarray, 
                   SkEn: np.ndarray) -> np.ndarray:
    """Compute Greens function

    Parameters
    ----------
    Hk_with_lr_energies : ndarray
        Hamiltonian with added self-energies of shape (N, N)
    SkEn : ndarray
        Array of Sk*En

    Returns
    -------
    ndarray
        Greens function of shape (N, N)
    """
    invG = SkEn - Hk_with_lr_energies
    return _inverse(invG)

def LDOS(G: np.ndarray) -> np.ndarray:
    """Compute Local Density of States using Greens function

    Parameters
    ----------
    G : ndarray
        Greens function of shape (N, N)

    Returns
    -------
    ndarray
        Array of shape (N,) for the LDOS of the N atoms/sites
    """
    return -(1/np.pi)*np.diag(G.imag)

def multi_LDOS(device: sisl.Geometry, electrode : sisl.Geometry, 
               lr_indices: tuple[np.ndarray, np.ndarray],
               *, energies: np.ndarray | list = [0], 
               Nk = 1, eta=1e-5) -> np.ndarray:
    """Compute the Local Density of States for each Energy and k-point

    Parameters
    ----------
    device : Geometry
        The device
    electrode : Geometry
        The electrode
    lr_indices : tuple[ndarray, ndarray]
        Arrays of shape (3, N) for the atom indices for the left and right electrodes 
    energies : ndarray  |  list, optional
        The input energy array of shape (M,), by default [0]
    Nk : int, optional
        The number of k-points around the Gamma-point, by default 1
    eta : complex, optional
        The small energy perturbation for use in self-energy calculations, by default 1e-5

    Returns
    -------
    ndarray
        Array of shape (Nk, len(energies), device.shape) for the LDOS calculations
    """
    k_direction = _direction(Nk=Nk, axis=1)
    Ne = len(energies)
    N_device = len(device)
    H_D = hamiltonian(device)
    kpts = sisl.MonkhorstPack(H_D, k_direction).k
    H_0 = hamiltonian(electrode)
    SE = RecursiveSI(H_0, infinite="+A")
    all_LDOS = np.zeros(shape=(Nk, Ne, N_device))
    for ik, kvec in enumerate(kpts):#, desc="k vecs"):
        Hk = H_D.Hk(k=kvec, format="array", dtype=complex)
        Sk = H_D.Sk(k=kvec, format="array", dtype=complex)
        for ie, E in tqdm(enumerate(energies), desc="Energies", total=len(energies)):
            En = E + 1j*eta
            SE_L, SE_R = lr_energies(electrode=SE, En=En, kvec=kvec)
            Hk = add_lr_energies(Hk, (SE_L, SE_R), lr_indices)
            G = compute_greens(Hk_with_lr_energies=Hk, SkEn=Sk*En)
            all_LDOS[ik, ie, ...] = LDOS(G)
    return all_LDOS