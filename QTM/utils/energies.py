import numpy as np
import sisl
from ._wrappers import timeit
from numba import njit
# from tqdm import tqdm

from typing import Sequence, Tuple
# from numpy.typing import NDArray
# from sisl.typing import GeometryLike

from sisl.physics import RecursiveSI

# from scipy.sparse.linalg import splu
# from scipy.sparse import isspmatrix_csc

# GPT parallelization
# import numpy as np
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.sparse.linalg import splu
from scipy.sparse import isspmatrix_csc
from scipy.sparse import identity
import os

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

# =============
# Self-energies
# =============
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


def LDOS_from_sparse(invG):
    """Compute Greens function from sparse matrix inputs.
    
    Paramters
    ---------
    invG : ndarray (sparse)
        Inverse of Greens function of shape (N,N)
     block_size : int
        Number of basis vectors to solve at once (tune this: 128-1024 typical).
    Returns
    -------
    ndarray
        The LDOS."""
    
    if not isspmatrix_csc(invG):
        invG = invG.tocsc()
    n = invG.shape[0]
    
    lu = splu(invG)
    diag = np.empty(n, dtype=complex)
    
    I = identity(invG.shape[0], format="csc")
    G_cols = lu.solve(I.toarray())  # solves for all e_i at once
    diag = np.diag(G_cols)
    
    return -np.imag(diag) / np.pi



def multi_LDOS(device: 'sisl.Geometry',
               electrode: 'sisl.Geometry',
               lr_indices: tuple[np.ndarray, np.ndarray],
               *,
               energies: np.ndarray | list = [0.0],
               Nk: int = 1,
               eta: float = 1e-5,
               form: str = "csc") -> np.ndarray:
    energies = np.asarray(energies)
    Ne = len(energies)
    k_direction = _direction(Nk=Nk, axis=1)
    
    H_D = hamiltonian(device)
    H_D.set_nsc((1,1,1))
    N_device = len(device)
    kpts = sisl.MonkhorstPack(H_D, k_direction).k
    
    H_0 = hamiltonian(electrode)
    SE = RecursiveSI(H_0, infinite="+A")
    all_LDOS = np.zeros(shape=(Nk, Ne, N_device), dtype=float)
    
    for ik, kvec in tqdm(enumerate(kpts), desc="k-points", leave=False):
        Hk = H_D.Hk(k=kvec, format=form, dtype=complex)
        Sk = H_D.Sk(k=kvec, format=form, dtype=complex)
        
        for ie, E in tqdm(enumerate(energies), desc="Energy"):
            En = E + 1j*eta
            SE_pair = lr_energies(electrode=SE, En=En, kvec=kvec)
            Hk_with_lr = add_lr_energies(Hk.copy(), SE_pair, lr_indices)
            invG = Sk*En - Hk_with_lr
            all_LDOS[ik, ie, ...] = LDOS_from_sparse(invG)
    return all_LDOS
    


################ PARALLELIZE ########################
# -------------------------
# Helper: diagonal via block solves
# -------------------------
def diag_of_inverse_sparse(invG_csc, block_size: int = 256) -> np.ndarray:
    """
    Compute diagonal of G = inv(invG_csc) without forming full inverse.
    Uses SuperLU factorization + block solves.
    """
    if not isspmatrix_csc(invG_csc):
        invG_csc = invG_csc.tocsc()

    n = invG_csc.shape[0]
    assert invG_csc.shape[0] == invG_csc.shape[1], "Matrix must be square"

    lu = splu(invG_csc)   # numeric factorization
    diag = np.empty(n, dtype=np.complex128)

    # Solve identity columns in blocks
    num_blocks = ceil(n / block_size)
    for b in range(num_blocks):
        start = b * block_size
        end = min(n, (b + 1) * block_size)
        m = end - start

        # Build RHS block: I[:, start:end] as dense array shape (n, m)
        B = np.zeros((n, m), dtype=np.complex128)
        B[start:end, np.arange(m)] = 1.0

        X = lu.solve(B)   # dense (n, m) result: columns of G
        diag[start:end] = np.diag(X[start:end, :])

    return diag
# -------------------------
# Worker: compute LDOS for one energy (called from threads)
# -------------------------
def _compute_ldos_with_precomputed_SE(E, Hk, Sk, SE_pair, lr_indices, eta, block_size):
    En = E + 1j * eta
    SE_L, SE_R = SE_pair
    Hk_with_lr = add_lr_energies(Hk.copy(), (SE_L, SE_R), lr_indices)
    invG = Sk * En - Hk_with_lr
    # choose between full-solve or block-solve inside diag_of_inverse_sparse
    return -np.imag(diag_of_inverse_sparse(invG, block_size=block_size)) / np.pi


# -------------------------
# Top-level: optimized multi_LDOS (parallel over energies per k-point)
# -------------------------
def multi_LDOS_parallel(device, electrode, lr_indices, *,
                              energies, Nk=1, eta=1e-5,
                              block_size=None, n_workers=None,
                              show_progress=True, form="csc",
                              full_solve_if_small=True):
    """
    Clean parallel-over-energies LDOS. Deterministic and easier to debug.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    energies = np.asarray(energies)
    Ne = len(energies)
    k_direction = _direction(Nk=Nk, axis=1)

    H_D = hamiltonian(device)
    N_device = len(device)
    kpts = sisl.MonkhorstPack(H_D, k_direction).k

    H_0 = hamiltonian(electrode)
    SE = sisl.physics.RecursiveSI(H_0, infinite="+A")

    all_LDOS = np.zeros((Nk, Ne, N_device), dtype=float)

    outer_iter = enumerate(kpts)
    if show_progress:
        outer_iter = tqdm(outer_iter, total=Nk, desc="k-points")

    for ik, kvec in outer_iter:
        Hk = H_D.Hk(k=kvec, format=form, dtype=complex)
        Sk = H_D.Sk(k=kvec, format=form, dtype=complex)
        n = Hk.shape[0]

        # auto block_size heuristics:
        if block_size is None:
            if full_solve_if_small and n <= 1400:
                block_size_local = n   # do all RHS at once
            else:
                block_size_local = min(512, max(64, n // 8))
        else:
            block_size_local = block_size

        # compute SEs serially for each energy (ensures determinism & thread-safety)
        se_pairs = [None] * Ne
        for ie, E in enumerate(energies):
            En = E + 1j * eta
            SE_L, SE_R = lr_energies(electrode=SE, En=En, kvec=kvec)
            se_pairs[ie] = (SE_L, SE_R)

        # parallel LU/solve (workers use precomputed SEs)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fut_to_idx = {}
            for ie, E in enumerate(energies):
                fut = executor.submit(_compute_ldos_with_precomputed_SE,
                                      E, Hk, Sk, se_pairs[ie], lr_indices,
                                      eta, block_size_local)
                fut_to_idx[fut] = ie

            if show_progress:
                pbar = tqdm(total=Ne, desc=f"energies (kpt {ik+1}/{Nk})", leave=False)
                for fut in as_completed(fut_to_idx):
                    ie = fut_to_idx[fut]
                    all_LDOS[ik, ie, :] = fut.result()
                    pbar.update(1)
                pbar.close()
            else:
                for fut in as_completed(fut_to_idx):
                    ie = fut_to_idx[fut]
                    all_LDOS[ik, ie, :] = fut.result()

    return all_LDOS