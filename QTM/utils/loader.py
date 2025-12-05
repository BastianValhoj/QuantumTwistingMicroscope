from pathlib import Path
from sisl.io import get_sile
import numpy as np

from tqdm.auto import tqdm


def path_finder(folder="QTM") -> Path:
    """Find the absolute path to project 'folder' (default 'QTM', the root dir)"""
    
    here = Path(__file__).resolve().parent
    
    for parent in [here] + list(here.parents):
        if parent.name == folder:
            return parent

def load_datastructure(LWC=(1,5,2), *, verbose=False) -> tuple:
    """Load data from folder with parameters L, W, C

    Parameters
    ----------
    LWC : tuple[int, int, int]
        The length width and center parameters to load data, by default (1,5,2) the smallest structure

    Returns
    -------
    electrode : sisl.Geometry
        
    ribbon : sisl.Geometry

    device : sisl.Geometry

    energies : np.ndarray
        numpy array of energies used for calculations
    ldos : np.ndarray
        numpy array of ldos. size is number of k-points, number of atoms, number of energies
    lr_idx : tuple[np.ndarray, np.ndarray]
        Respectively, the atom index of left and right electrodes. Each array is of shape (3, M) for the M atoms in each of the 3 left/right electrodes
    """
    PROJECT_ROOT = path_finder(folder="QTM")
    results_folder = PROJECT_ROOT / "results"
    l, w, c = LWC
    params_path = results_folder / f"L{l}_W{w:02}_C{c:02}"
    DATA = np.load(f"{params_path}/calculations.npz")
    electrode = get_sile(f"{params_path}/electrode.xyz").read_geometry()
    ribbon = get_sile(f"{params_path}/ribbon.xyz").read_geometry()
    device = get_sile(f"{params_path}/device.xyz").read_geometry()
    if verbose:
        tqdm.write(f"params : (L, W, C) = {LWC}")
    return electrode, ribbon, device, DATA
    