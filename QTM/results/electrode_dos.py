
from utils import hamiltonian, load_datastructure
import sisl
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm.auto import tqdm

# directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

candidates = [SCRIPT_DIR] + list(SCRIPT_DIR.parents)

PROJECT_ROOT = None
for p in candidates:
    if p.name == "QTM":
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Could not find project root 'QTM' starting from {SCRIPT_DIR}")
    
BASE_DIR = PROJECT_ROOT / "results" # QTM/results


def electrode_dos(ham, nk, energies):
    """compute DOS from built-ins
    
    Parameters
    ----------
    ham : Hamiltonian of shape (K, K)
        Hamiltonian in question 
    nk : int
        number of k points
    energies : ndarray of shape (N,)
        list of energies to compute the dos

    Returns
    -------
    dos : ndarray of shape (N,)
        density of state
    bs : ndarray of shape (M, N)
        bandstructure eigenvalues 
    lk : ndarray of shape (M,)
        scaled kpoints to linear values for use in plotting
    """
    dist = sisl.get_distribution(method="lorentzian", smearing=0.05)
    
    bz = sisl.MonkhorstPack(ham, [nk, 1, 1])
    kvecs = [[0,0,0], [0.5, 0, 0]]
    bands = sisl.BandStructure(ham, kvecs, 15, [r"$\Gamma$", "$K$"])
    
    lk, kticks, knames = bands.lineark(ticks=True)
    bs = bands.apply.array.eigh()
    dos = (bz.apply.average).eigenstate(wrap = lambda x: x.DOS(energies, distribution=dist)) 
    return dos, bs, (lk, kticks, knames)

def plot_dos_band(electrode, nk, energies, LWC):
    L, W, C = LWC
    """Plot the device band structure and DOS"""
    H = hamiltonian(electrode)   
    
    dos, bands, k = electrode_dos(H, nk, energies)
    fig, axes = plt.subplots(1,2, sharey=True)
    fig.suptitle(f"Electrode L={L}, W={W}, C={C}")
    axes[0].set_title("Band structure")    
    axes[0].plot(k[0], bands)    
    axes[0].set_xlabel("$k$", size=15)
    axes[0].set_xticks(k[1])
    axes[0].set_xticklabels(k[2])
    
    axes[1].set_title("DOS")
    axes[1].plot(dos, energies, label="built-in DOS")
    axes[1].set_xlabel("DOS", size=15)
    
    axes[0].set_ylabel("E", size=15)
    fig.tight_layout()
    for ax in axes:
        ax.grid()
    
    return fig, axes


def parse_LWC(folder_name: str):
    parts = folder_name.split("_")
    L = int(parts[0][1:]) # remove 'L'
    W = int(parts[1][1:]) # remove 'W'
    C = int(parts[2][1:]) # remove 'C'
    return (L, W, C)
def process_folder(folder: Path):
    LWC = parse_LWC(folder.name)
    electrode, ribbon, device, DATA = load_datastructure(LWC)
    
    fig, ax = plot_dos_band(electrode, 100, DATA["energies"], LWC)
    minmax = 2
    ax[0].set(ylim=(-minmax, minmax))
    fig.savefig(folder / "electrode_bands_dos.png")
    fig.savefig(folder / "electrode_bands_dos.svg")
    plt.close()
    
    
    
    
def main():
    all_folders = sorted(BASE_DIR.glob("L*_W*_C*"))
    for folder in tqdm(all_folders, desc="Processing folders", total=len(all_folders)):
        if folder.is_dir():
            process_folder(folder)
    



if __name__ == "__main__":
    main()