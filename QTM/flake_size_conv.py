import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sisl.geom import graphene_flake

from utils import hamiltonian

from pathlib import Path

ENERGIES = np.linspace(-2, 2, num=50)
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLE = [":", "-.", "--", "-"]
ALPHA = [0.9, 0.8, 0.7, 0.6]
LINEWIDTH = [1.6, 1.8, 2.0, 2.5]
FLAKE_SHELLS = [10, 15, 20, 25]
NSC = [1,1,1]


# find QTM/figure directory and create it if it doesn't exist
SCRIPT_DIR = Path(__file__).resolve().parent
candidates = [SCRIPT_DIR] + list(SCRIPT_DIR.parents)

PROJECT_ROOT = None
for p in candidates:
    if p.name == "QTM":
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Could not find project root 'QTM' starting from {SCRIPT_DIR}")
    
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)





def _calc(shells):
    flake = graphene_flake(shells=shells)
    H = hamiltonian(flake)
    H.set_nsc(NSC)
    # print(f"{H.nsc = }")
    es = H.eigenstate()
    pdos = es.PDOS(E=ENERGIES).squeeze() # reduce from (1,M,N) -> (M,N)
    center_pdos = pdos[0]
    return center_pdos

def main():
    fig, axes = plt.subplots(figsize=(8,6), dpi=100)
    for i, s in tqdm(enumerate(FLAKE_SHELLS), desc=f"Shells iteration", total=len(FLAKE_SHELLS)):
        x, y = _calc(s), ENERGIES
        
        axes.plot(x, y, label=f"{s = }",
                  alpha=ALPHA[i],
                  linestyle=LINESTYLE[i],
                  lw=LINEWIDTH[i]
        )
        # break
    axes.set_xlabel("PDOS (center atom)", size=14)
    axes.set_ylabel("Energies", size=14)
    legend = axes.legend(fontsize=14, title="Flake shell size")
    legend.get_title().set_fontsize(16)
    fig.suptitle("Flake size PDOS conv.", size=20)
    fig.savefig(FIG_DIR / "flake_size_conv.svg")
    plt.close()





if __name__ == "__main__":
    main()