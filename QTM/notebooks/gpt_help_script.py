import numpy as np
import sisl
from scipy.spatial import cKDTree
from sisl.physics import RecursiveSI
from numba import njit

# ===================
### Create nanoribbon
# ===================
def _reorder_atoms(structure):
    x, y, z = structure.xyz.T
    order = np.lexsort((y, z, x))
    return structure.sub(order) # 'use only these atoms' will then redefine the ordering of atoms to some easily predictable way

def make_nanoribbon(width, length, **kwargs):
    """Generate a graphene nanoribbon."""
    bond = kwargs.get('bond', 1.42)
    kind = kwargs.get('kind', 'armchair')
    BC = kwargs.get("BC", True)
    vacuum = kwargs.get('vacuum', 3.0)
    ribbon = sisl.geom.graphene_nanoribbon(width=width, bond=bond, kind=kind, vacuum=vacuum)
    ribbon = ribbon.repeat(length, axis=0)
    ribbon = _reorder_atoms(ribbon)
    if not BC:
        ribbon.set_nsc((1,1,1)) # avoid periodicity in transport direction
    # else: ribbon.set_nsc((3,1,1)) # default behavior
    return ribbon

def make_device_nanoribbon(electrode: sisl.Geometry, center_size : int):
    assert center_size > 0, "must have a center size of at least 1 `electrode`"
    return electrode.tile(2+center_size, 0)


def _device_atoms(device, electrode, reset=False):
    """Change the left electrode and right electrode atoms"""
    if not reset:
        N = len(electrode)
        device.atoms[:N] = sisl.Atoms("N") # set left electrode atoms to N (blue)
        device.atoms[-N:] = sisl.Atoms("O") # set right electrode atoms to O (red)
    
    if reset:
        device.atoms[:] = sisl.Atoms("C") # set all atoms to C

def _set_electrode_atoms(device, electrode):
    _device_atoms(device, electrode, reset=False)

def _return_device(device, electrode):
    _device_atoms(device, electrode, reset=True)
    

def find_electrode_indices(device, num_electrodes):
    symbols = np.array([atom.symbol for atom in device.atoms])
    
    # oxygen corresponds to right electrode (red)
    idx_O = np.where(symbols == "O")[0]
    NO = len(idx_O) // num_electrodes # number of O atoms per electrode
    
    # nitrogen correpsonds to left electrode (blue)
    idx_N = np.where(symbols == "N")[0]
    NN = len(idx_N) // num_electrodes # number of N atoms per electrode
    
    left_electrodes = []
    right_electrodes = []
    for i in range(num_electrodes):
        left = idx_N[i*NN:(i+1)*NN]
        right = idx_O[i*NO:(i+1)*NO]
        left_electrodes.append(left)
        right_electrodes.append(right)
    return np.array(left_electrodes), np.array(right_electrodes)

def get_coordinates(structure):
    """Return atomic coordinates as numpy array."""
    if isinstance(structure, sisl.Geometry):
        return structure.xyz
    elif isinstance(structure, np.ndarray):
        return structure
    else:
        raise TypeError("Input must be a sisl.Geometry or numpy.ndarray.")

def find_nearest_atoms(structure, center, n=6):
    """Return indices of n atoms closest to a given center."""
    coords = get_coordinates(structure)
    distances = np.linalg.norm(coords - center, axis=1)
    return np.argsort(distances)[:n]

def find_overlap(structure : sisl.Geometry, tol=0.1):
    """Find overlapping atoms in a structure."""
    coords = get_coordinates(structure)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=tol)  # find pairs of atoms closer than `tol`
    # print(f"{type(pairs) = }")
    return pairs

def delete_overlaps(structure : sisl.Geometry, pairs : set | None = None):
    """Delete overlapping atoms from a structure."""
    if pairs is None:
        overlaps = find_overlap(structure)
    else:
        overlaps = pairs
    to_delete = set()
    for i, j in overlaps:
        to_delete.add(j)  # arbitrarily delete the second atom in each pair
    mask = np.array([i not in to_delete for i in range(len(structure))])
    return structure.sub(mask)

def guess_hexagon_center(structure):
    """
    Try to locate the center of a hexagon near the geometric center.
    If not found, iteratively shift the center guess.
    """
    BOND = 1.42
    NN_2_DIST = 2 * BOND * np.cos(np.deg2rad(30))  # distance between two second-nearest neighbors in graphene
    RTOL = 0.05  # relative tolerance for checking if center lies on bonds
    coords = get_coordinates(structure)
    center = coords.mean(axis=0) 
    
    atom_idx = find_nearest_atoms(structure, center, n=6)
    hex_coords = coords[atom_idx]
    center = hex_coords.mean(axis=0)
    atom1, atom2 = hex_coords[[0, 1]]
    if np.isclose(atom1[1], atom2[1], rtol=RTOL) and np.isclose(atom1[1], center[1], rtol=RTOL):
        print("Warning: The geometric center lies on bonds. Trying to shift the center by half atomic distance to 2. NN.")
        print(f"{'Atom 1':>20}:", atom1)
        print(f"{'Atom 2':>20}:", atom2)
        center += np.array([0, NN_2_DIST/2, 0])  # shift the center by half the distance to the 2nd NN
        print(f"{'New Center':>20}:", center)

    return atom_idx, center
    
def structure(nanoribbon: sisl.Geometry, electrode: sisl.geometry, *, repeat=3):
    assert repeat in [1, 2, 3], "repeat can only be 1, 2, or 3"
    _, origin_of_rotation = guess_hexagon_center(nanoribbon)
    axis_of_rotation = [0,0,1]
    
    
    RIBBON = nanoribbon.copy()
    _set_electrode_atoms(RIBBON, electrode)
    FINAL = RIBBON.copy()
    for i in range(1, repeat):
        angle = 60
        angle = angle*(-1) if i%2 == 0 else angle
        rotated_nanoribbon = RIBBON.rotate(angle=angle, v=axis_of_rotation, origin=origin_of_rotation)
        FINAL += rotated_nanoribbon
    
    # remove overlapping atoms
    FINAL = delete_overlaps(FINAL)
    
    # find the indices for the electrodes using the different atom species
    left_indices, right_indices = find_electrode_indices(FINAL, repeat)
    
    # return to original atom species -- Carbon -- for all atoms
    _return_device(FINAL, electrode)
    
    return FINAL, left_indices, right_indices

# ===================
### compute self energies
# ===================

def _direction(**kwargs):
    "infer direction of extending kvector"
    Nk = kwargs.get("Nk", 1)
    axis = kwargs.get("axis", 1)
    if isinstance(axis, int):
        if not axis in range(3): # 0, 1, or 2
            raise ValueError("'axis' must be  0,  1,  or  2.")
        d = [1, 1, 1]
        d[axis] = Nk
        return d, Nk
    else:
        raise ValueError("axis must be  int.")
        
@njit
def hermconj(matrix):
    "Hermitian conjugate of 2d matrix"
    assert matrix.ndim == 2, "matrix must be 2D"
    return matrix.T.conj()

@njit
def calc_gamma(se):
    "Compute left/right broadening matrix from left/right self-energy"
    return 1j*(se - hermconj(se))


@njit
def greens(left, right, E, H):
    "compute freens function from Energy, device hamiltonian and left/right self-energies."
    assert left.shape == right.shape, "left and right self-energies not identical."
    N = len(left)
    invG = E - H
    invG[ :N,  :N] -= left
    invG[-N:, -N:] -= right
    return np.linalg.inv(invG)

@njit
def spectral(Greens, Gamma):
    "compute left/right spectral function from greens function and left/right broadening matrix"
    return Greens @ Gamma @ hermconj(Greens)

@njit
def _T(AR, GammaL):
    return np.trace(AR @ GammaL)


def lr_energies(device, electrode, **kwargs):
    energies = kwargs.get("energies", 0)
    eta = kwargs.get("eta", 1e-5)
    if isinstance(energies, (float, int)): # convert single value energies to list 
        energies = [energies]
        
    
    NE = len(energies)
    N_device = len(device)
    N_electrode = len(electrode)
    k_direction, Nk = _direction(**kwargs)
    kpts = sisl.MonkhorstPack(device, k_direction).k
    kwargs.setdefault("kpts", kpts)
    kwargs.setdefault("energies", energies)
    SE = RecursiveSI(electrode, "+A", eta=eta)
    # greens_kE = np.zeros(shape=(Nk, NE, N_device, N_device), dtype=complex)
    left_energies = np.empty(shape=(Nk, NE, N_electrode, N_electrode), dtype=complex)
    right_energies = left_energies.copy()
    for iter_k, kvec in enumerate(kpts):
        # Hk = device.Hk(k=kvec, format="array").astype(complex)
        # Sk = device.Sk(k=kvec, format="array").astype(complex)
        
        for iter_E, E in enumerate(energies):
            En = E + 1j*eta
            SE_L, SE_R = SE.self_energy_lr(E=En)
            # greens_kE[iter_k,iter_E, ...] = greens(left=SE_L, right=SE_R, E=En*Sk, H=Hk)
            left_energies[iter_k, iter_E, ...] = SE_L
            right_energies[iter_k, iter_E, ...] = SE_R
    return left_energies, right_energies, kwargs

def hamiltonian(structure, **kwargs):
    """Create a tight-binding Hamiltonian for a given structure."""
    r = kwargs.get("r", (0, 1.44)) # bond length in Angstrom
    t = kwargs.get("t", (0.0, -2.7)) # energy in eV
    H = sisl.Hamiltonian(structure)
    H.construct([r, t])
    if kwargs.get("finalize", False):
        H.finalize() # make hamiltonian sparse
    return H

def device_hamiltonian(device, ribbon, electrode, left_idx, right_idx, **kwargs):
    ribbon_HAM = hamiltonian(ribbon)
    electrode_HAM = hamiltonian(electrode)
    device_HAM = hamiltonian(device)
    device_HAM.set_nsc((1,1,1)) # no PBC for structure
    
    left_energies, right_energies, kwargs = lr_energies(ribbon_HAM, electrode_HAM, **kwargs)
    assert left_energies.shape == right_energies.shape, "dimensions of left and right energies does not mathc..."
    device = device.copy()
    # electrode = electrode.copy()
    
    num_k, num_E = left_energies.shape[:2]
    hams = np.zeros(shape=(num_k, num_E, *device_HAM.Hk(format="array").shape), dtype=complex)
    for iter_k, kvec in enumerate(kwargs.get("kpts")):
        Hk = device_HAM.Hk(k=kvec, format="array").astype(complex)
        for iter_E, E in enumerate(kwargs.get("energies")):
            for arm, (iL, iR) in enumerate(zip(left_idx, right_idx)):
                Hk[None, None, iL[0]:iL[-1]+1, iL[0]:iL[-1]+1] += left_energies
                Hk[None, None, iR[0]:iR[-1]+1, iR[0]:iR[-1]+1] += right_energies
            hams[iter_k, iter_E, ...] = Hk
    
    return hams

def main(WIDTH, LENGTH):
    electrode = make_nanoribbon(WIDTH, LENGTH)
    ribbon = make_device_nanoribbon(electrode, 1)
    device, left_idx, right_idx = structure(ribbon, electrode, repeat=3)
    HAMS = device_hamiltonian(device, ribbon, electrode, left_idx, right_idx)
    return HAMS

if __name__ == "__main__":
    WIDTH = 5
    LENGTH = 2
    H = main(WIDTH, LENGTH)
    print("DONE!")
    