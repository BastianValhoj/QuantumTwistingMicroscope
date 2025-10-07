import numpy as np
import sisl
from time import time
from scipy.spatial import cKDTree

## Wrappers
from ._wrappers import timeit, count_removals


# ---------------------------
# Geometry helper functions
# ---------------------------
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

# ---------------------------
# Structure generation
# ---------------------------
def find_overlap(structure : sisl.Geometry, tol=0.1):
    """Find overlapping atoms in a structure."""
    coords = get_coordinates(structure)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=tol)  # find pairs of atoms closer than `tol`
    # print(f"{type(pairs) = }")
    return pairs

@count_removals
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

def set_index(overlaps : set, indexes : dict):
    """Set atom indexes for each arm after removing overlaps.
    
    Args:
        overlaps (set): Set of tuples representing overlapping atom pairs.
        indexes (dict): Dictionary with initial atom indexes for each arm.
    Returns:
        dict: Updated dictionary with atom indexes for each arm after removing overlaps.
    
    """    
    # remove overlaps from arms 1 and 2
    for _, b in overlaps: # always remove the second atom in the pair
        for arm in list(indexes.keys())[1:]: # arm 0 is the base ribbon, so skip it
            if b in indexes[arm]: # if the atom is in the current arm
                indexes[arm].remove(b) # remove it
    
    arm_lengths = map(len, indexes.values()) # lengths of each arm after removing overlaps
    cummulative = np.cumsum([0, *arm_lengths]) # cumulative sum to get new indexes
    
    for key in indexes: # reassign indexes based on new lengths
        indexes[key] = list(range(cummulative[key], cummulative[key + 1])) 
    return indexes

def _init_indexes(nanoribbon):
    """Initialize atom indexes for each arm of the structure."""
    indexes = {} # dictionary to hold indexes for each arm
    total_atoms = len(nanoribbon) # total number of atoms in the structure
    arm_length = total_atoms // 3 # assuming 3 arms of equal length
    for i in range(3):
        indexes[i] = list(range(i*arm_length, (i+1)*arm_length))
    return indexes


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

@timeit
def generate_structure(width=5, length=8, **kwargs):
    """Generate 3 rotated nanoribbons stacked around a central hexagon."""

    base = make_nanoribbon(width=width, length=length, **kwargs)
    
    # find central hexagon
    _, rotation_origin = guess_hexagon_center(base)
    
    # rotate and add ribbons
    structure = base.copy()
    for i in range(1, 3):
        angle = i * 60
        rotated = base.rotate(angle=angle, v=[0, 0, 1], origin=rotation_origin)
        structure += rotated
        
    overlaps = find_overlap(structure)
    
    if overlaps:
        indexes = _init_indexes(structure)
        indexes = set_index(overlaps, indexes)
        structure = delete_overlaps(structure)
        return structure, indexes
    else:
        return structure, indexes

