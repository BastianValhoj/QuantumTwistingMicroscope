from utils.structure import build_electrode, build_nanoribbon, build_reduced_device, infer_centersize
from utils.energies import hamiltonian, multi_LDOS

from tqdm.auto import tqdm
import sisl
import numpy as np

import os
from pathlib import Path

from itertools import product

Nk = 1
ENERGIES = np.linspace(-2, 2, num=100)
C_PARMS = [None, 0.05, 0.5, 1, 5, 10, 20]

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

def main(LWC_combinations):
    for L, W, C in tqdm(LWC_combinations, desc="LWC combinations", total=len(LWC_combinations), leave=True):
        folder = BASE_DIR / f"L{L}_W{W:02}_C{C:02}"
        folder.mkdir(parents=True, exist_ok=True) # create parameter folder - also create parents necessary
        tqdm.write(f"Saving data to folder : {folder}")
        
        # Build structures
        electrode = build_electrode(width=W, length=L)
        ribbon = build_nanoribbon(electrode=electrode, center_size=C)
        device, lr_idx = build_reduced_device(ribbon, electrode, repeat=3)
        
        # Save structures
        electrode.write(folder / "electrode.xyz")
        ribbon.write(folder / "ribbon.xyz")
        device.write(folder / "device.xyz")
        
        ldos_by_C = {}
        for modC in tqdm(C_PARMS, desc="Modulation parameter", leave=False):
            # print(f"  -> Computing LDOS for modulated_SE={modC}")
            ldos_by_C[modC] = multi_LDOS(device=device,
                                         electrode=electrode,
                                         lr_indices=lr_idx,
                                         energies=ENERGIES,
                                         Nk=Nk, eta=1e-5, form="csc",
                                         modulate_SE=modC,
                                         )

        save_dict = {
            "energies": ENERGIES,
            "lr_idx": lr_idx,
        }
        
        # Insert each LDOS into the dict under a clear key
        for modC, arr in ldos_by_C.items():
            if modC is None:
                key = "ldos"
            else:
                key = f"ldos_C{modC:g}" # safe, reable key
            save_dict[key] = arr
            
            
        np.savez(folder / "calculations", **save_dict)
        
        pass



if __name__ == "__main__":
    Llist = range(1,3)
    ilist = range(0,8)
    all_inputs = product(Llist, ilist)
    unique_combinations = set(infer_centersize(l, i) for l, i in all_inputs)
    
    # Sort by the 2nd element (index 1) only influencing the order of computation.
    sorted_combinations = sorted(unique_combinations, key=lambda t: t[1])
    # print(sorted_combinations)
    main(sorted_combinations)
    
