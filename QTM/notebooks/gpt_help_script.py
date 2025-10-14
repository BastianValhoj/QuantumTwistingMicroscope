import sisl
import numpy as np
from sisl.physics import RecursiveSI, SemiInfinite

def _reorder_atoms(structure):
    x, y, z = structure.xyz.T
    order = np.lexsort((x, y, z))
    return structure.sub(order) # 'use only these atoms' will then redefine the ordering of atoms to some easily predictable way

def make_nanoribbon(width, length, **kwargs):
    """Generate a graphene nanoribbon."""
    bond = kwargs.get('bond', 1.42)
    kind = kwargs.get('kind', 'armchair')
    no_BC = kwargs.get("BC", False)
    vacuum = kwargs.get('vacuum', 3.0)
    ribbon = sisl.geom.graphene_nanoribbon(width=width, bond=bond, kind=kind, vacuum=vacuum)
    ribbon = ribbon.repeat(length, axis=0)
    ribbon = _reorder_atoms(ribbon)
    if no_BC:
        ribbon.set_nsc((1,1,1)) # avoid periodicity in transport direction
    # else: ribbon.set_nsc((3,1,1)) # default behavior
    return ribbon


def terminal(width, length, depth) -> list[float]:
    """From the width and length return the atom indices highligting the terminal atoms of depth `d`"""
    atom_idx = []
    for i in range(1, width+1):
        for d in range(depth):
            atom_idx.append(2*i*length - (1+d))
    return atom_idx

def main(E):
    WIDTH = 10
    LENGTH = 15
    DEPTH = 4
    
    device = make_nanoribbon(width=WIDTH, length=LENGTH-2*DEPTH)
    lead = make_nanoribbon(width=WIDTH, length=DEPTH)
    
    r = (0.1, 1.5)
    t = (0.0, -2.7)

    H_device = sisl.Hamiltonian(device)
    H_lead = sisl.Hamiltonian(lead)


    H_device.construct([r, t])
    H_lead.construct([r,t]) 
    
    directions = ["+A", "-A"]
    
    leadSE = RecursiveSI(spgeom=H_lead, infinite=directions[0])
    
    return leadSE.self_energy(E)


if __name__ == "__main__":
    SE = main(E=0)
