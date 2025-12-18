# base imports
import numpy as np
import sisl
from scipy.spatial import cKDTree

# custom imports
from ._wrappers import count_removals
from .helpers import in_notebook

# debugging
from warnings import warn


from tqdm.auto import tqdm

# ---------------------------
# Geometry helper functions
# ---------------------------
def get_coordinates(structure: sisl.Geometry | np.ndarray) -> np.ndarray:
    """
    Return atomic coordinates from a sisl Geometry or NumPy array.

    Parameters
    ----------
    structure : Geometry or ndarray
        Either a `sisl.Geometry` object or a 2D NumPy array of shape (N, 3).

    Returns
    -------
    ndarray
        Coordinates as an array of shape (N, 3).

    Raises
    ------
    TypeError
        If `structure` is not a valid type.
    ValueError
        If a NumPy array is not of shape (N, 3).
    """
    if isinstance(structure, sisl.Geometry):
        return structure.xyz
    elif isinstance(structure, np.ndarray):
        if structure.shape[-1] != 3:
            raise ValueError("If `structure` is a ndarray, it must be of shape (N, 3)")
        return structure
    else:
        raise TypeError("Input must be ndarray or Geometry")

def _reorder_atoms(structure: sisl.Geometry) -> sisl.Geometry:
    """Order atoms with the propagation direction (x) last

    Parameters
    ----------
    structure : Geometry
        Structure to order  

    Returns
    -------
    Geometry
        The ordered atoms

    Raises
    ------
    ValueError
        If wrong input type is used
    """
    if not isinstance(structure, sisl.Geometry):
        raise ValueError(f"_reoder_atoms : argument type must be Geometry, was {type(structure)}")
    x, y, z = get_coordinates(structure).T
    order = np.lexsort((y,z,x))
    return structure.sub(order)

def find_nearest_atoms(coords: np.ndarray, point: np.ndarray, 
                       neighbours=6) -> np.ndarray:
    """Return indices of nearest atoms to point

    Parameters
    ----------
    coords : ndarray
        xyz array of shape (N, 3) for atoms
    point : ndarray
        xyz array of shape (3,) for a point of interest
    neighbours : int, optional
        number of neighbouring atoms to find, by default 6
    
    Returns
    -------
    ndarray
        Array of shape (neighbours,) for atom indices for nearest atoms to `point`
    """
    
    distances = np.linalg.norm(coords - point, axis=1)
    output = np.argsort(distances)[:neighbours]
    if len(output) == 1:
        output = output[0]
    return output

def guess_hexagon_center(structure: sisl.Geometry) -> np.ndarray:
    """For use for determining center of rotation. 
    Find the hexagon used for center of rotation by use of geometric center

    Parameters
    ----------
    structure : Geometry
        Geometry to find the center of

    Returns
    -------
    ndarray of shape (3,)
        xyz coordinates of center for central (rotation) hexagon
    """
    bond = 1.42 # [Å] distance to nearest neighbour
    # distance_2_nn = 2*bond*np.cos(np.deg2rad(30)) # [Å] distance to the second nearest neighbour
    dy = bond*np.cos(np.deg2rad(30)) # [Å] half distance to the second nearest neighbour
    atol = 5e-2 # tolerance for determining of values are close
    coords = get_coordinates(structure)
    geom_center = coords.mean(axis=0)
    atom_idx = find_nearest_atoms(coords, geom_center, neighbours=6)
    hex_coords = coords[atom_idx]
    hex_center = hex_coords.mean(axis=0)
    
    dist_to_atoms = np.linalg.norm(hex_coords - hex_center[None, :], axis=1) # proposed hex center distance to corresponding hex atoms
    # check if hex center (the rotation center) is different from the geom center,
    # or if hex center distance is not bond length (should be for hexagon geometry)
    if not np.all(np.isclose(hex_center, geom_center, atol=atol)) or not np.all(np.isclose(dist_to_atoms, bond, atol=atol)):
        hex_center += np.array([0, dy, 0]) # make a slight shift in y to find new hex 
        warn_string = "\n" \
        "### Warning: The center of ration is on the bond between atoms.\n" \
        f"### {'Hex Center':>20}: {hex_center}\n" \
        # f"### {'Geometric Center':>20}: {geom_center}"
        f"### adding shift in y {dy = } and find new hex center"
        warn(warn_string)
        
        # use the shifted hex_center to find new hex_center
        atom_idx = find_nearest_atoms(coords, hex_center, neighbours=6)
        hex_coords = coords[atom_idx]
        hex_center.mean(axis=0)
    
    return hex_center

# ================================================
# Helper functions for determining electrode atoms
# ================================================
def _mark_electrode(device: sisl.Geometry, electrode: sisl.Geometry):
    """Temporarily mark left/right electrodes for index finding.
    Mutates the input device.
    
    Parameters
    ----------
    device : Geometry
        The device to have the electrode marked
    electrode : Geometry
        Used for determining the atomic indices to mark
    """
    N = len(electrode)
    device.atoms[:N] = sisl.Atom("N")
    device.atoms[-N:] = sisl.Atom("O")
    
def _reset_atoms(device: sisl.Geometry): 
    """Reset all device atoms to carbon.
    Mutates the input device
    
    Parameters
    ----------
    device : Geometry
        The device to reset
    """
    device.atoms[:] = sisl.Atom("C")
    
def find_electrode_indices(device: sisl.Geometry, num_electrodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of left/right electrodes for each nanoribbon arm.

    Parameters
    ----------
    device : Geometry
        The device to investigate
    num_electrodes : int
        Number of left and right electrodes (usually set to 3)

    Returns
    -------
    tuple[ndarray, ndarray]
        Indices for left and right electrode atoms. Number of rows equals `num_electrodes`
    """
    symbols = np.array([atom.symbol for atom in device.atoms])
    idx_N = np.where(symbols == "N")[0]
    idx_O = np.where(symbols == "O")[0]
    NN = len(idx_N) // num_electrodes
    NO = len(idx_O) // num_electrodes
    lefts, rights = [], []
    for i in range(num_electrodes):
        lefts.append(idx_N[i*NN:(i+1)*NN])
        rights.append(idx_O[i*NO:(i+1)*NO])
    return np.array(lefts), np.array(rights)

# ==================================
# Overlap identification and removal
# ==================================
def infer_centersize(l, i):
    """Given L and a starting i, find the smallest `i' >=  i` such that `C = 2*(1+i)/l` is a positive integer.
    
    Parameters
    ---
    L : int
        The integer for the length of electrode
    i : int
        determine width by `W = 5+4*i`
    Returns
    ---
    tuple : (L, W, C)
    """
    if (not isinstance(l, int)) or (not isinstance(i, int)):
        raise TypeError("l and i must be integer")
    if l < 1:
        raise ValueError("l must be at least 1")
    if i < 0:
        raise ValueError("i must be non-negative")
    
    # we want  L | (2*(i+1))  meaning 2(1+i) is divisible by L yielding an integer.
    # ("|" means that for an integer k: 2*(i+1) = kL)
    
    # so 2*(1+i) = 0 mod L -> i = -1 mod (L/gcd(L, 2))
    import math
    g = math.gcd(l, 2)
    spacing = l // g # modulus for i
    
    # solve for i = -1 (mod s), but Python want positives
    target = (-1) % spacing # the smallest non-negative integer satisfying the modular equation
    if i <= target: # if i is too small, just set to target
        i = target
    else: # increase i to next compatible value
        # if in_notebook():
        #     tqdm.write(f"{(l, i) = } are incompatbile. Increasing i.")
        i = i + ((target - i) % spacing) 
    
    w = 5 + i*4
    c = 2*(1+i)//l # integer
    return l, w, c
    
# ==================================
# Overlap identification and removal
# ==================================
def find_overlap(structure: sisl.Geometry) -> set[tuple[int, int]]:
    """pairs of atoms that overlap in position

    Parameters
    ----------
    structure : Geometry
        The structure to check for overlaps

    Returns
    -------
    set[tuple[int, int]]
        A set of tuples. The tuples are the atomic indices of atoms having the same xyz
    """
    OVERLAP_TOLERANCE = 0.1
    coords = get_coordinates(structure)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=OVERLAP_TOLERANCE)
    return pairs

@count_removals
def remove_overlaps(structure: sisl.Geometry, 
                    pairs: set[tuple[int, int]] | None = None) -> sisl.Geometry:
    """Delete overlapping atoms from a structure.

    Parameters
    ----------
    structure : Geometry
        Structure to find and delete overlaps
    pairs : `set  |  None`, optional
        (If None find the pairs first and) set of atom indices having the same xyz, by default None

    Returns
    -------
    Geometry
        The structure with the overlapping atoms removed.
    """
    if pairs is None:
        overlaps = find_overlap(structure)
    else:
        overlaps = pairs
    
    to_delete = set()
    for i, j in overlaps:
        to_delete.add(j)  # arbitrarily delete the second atom in each pair
    mask = np.array([i not in to_delete for i in range(len(structure))])
    return structure.sub(mask)

# ================================================
# Create electrode and device "arm" (a nanoribbon)
# ================================================
def build_electrode(width: int, length: int, 
                    kind="armchair") -> sisl.Geometry:
    """Build electrode as a nanoribbon

    Parameters
    ----------
    width : int
        The number of atoms for the width of arms of final device
    length : int
        The length of the electrode (and thereby the length of the device "arms")
    kind : str, optional
        The type of nanoribbon, by default "armchair"

    Returns
    -------
    Geometry
        The final electrode
    """
    vac = 3
    electrode = sisl.geom.graphene_nanoribbon(width, kind=kind, vacuum=vac)
    electrode = electrode.tile(length, axis=0)
    electrode = _reorder_atoms(electrode)
    return electrode

def build_nanoribbon(electrode: sisl.Geometry, 
                     center_size : int) -> sisl.Geometry:
    """Make nanoribbon by tiling the electrode based on `center_size`

    Parameters
    ----------
    electrode : Geometry
        Use electrode as base for device
    center_size : int
        How many `electrode` to use for the center of device

    Returns
    -------
    Geometry
        Nanoribbon creating by tiling the electrode
    
    Raises
    ------
    ValueError
        If `center_size is not a positive integer
    """
    if not isinstance(center_size, int) or center_size <= 0:
        raise ValueError("`center_size` must be a positive integer")
    ribbon = electrode.tile(2+center_size, 0)
    # ribbon.set_nsc((1,1,1))
    return ribbon

# ==================================
# Build actual structure of interest
# ==================================
def build_reduced_device(nanoribbon: sisl.Geometry, electrode: sisl.Geometry, 
                         *, repeat=3) -> tuple[sisl.Geometry, np.ndarray, np.ndarray]:
    """Build the reduced structure device along with indicis of left and right electrodes for each nanoribbon arm

    Parameters
    ----------
    nanoribbon : Geometry
        Ribbon to repeat, rotate and add to the final structure
    electrode : Geometry
        The structure that is used for both left and right electrodes
    repeat : int, optional
        Determines how many "arms" the structure should have, by default 3

    Returns
    -------
    Geometry
        The device 
    tuple[ndarray, ndarray]
        arrays of atomic indices for left and right electrodes each of shape (repeat, N)
    
    Raises
    ------
    ValueError
        If `repeat` is not either 1, 2, or 3
    """
    if not repeat in [1,2,3]:
        raise ValueError("`repeat` must be either 1, 2, or 3")
    nanoribbon = nanoribbon.copy()
    
    origin_of_rotation = guess_hexagon_center(nanoribbon)
    rotation_axis = [0,0,1] # rotate around Z
    _mark_electrode(nanoribbon, electrode)
    
    device = nanoribbon.copy()
    for i in range(1, repeat):
        angle = 60 # degrees
        angle = angle*(-1) if i%2 == 0 else angle
        rotated_nanoribbon = nanoribbon.rotate(angle=angle, v=rotation_axis, origin=origin_of_rotation)
        device += rotated_nanoribbon
    
    device = remove_overlaps(device)
    left_indices, right_indices = find_electrode_indices(device, repeat)
    _reset_atoms(device)
    device.set_nsc((1,1,1))
    
    return device, (left_indices, right_indices)



