from utils.structure import build_electrode, build_nanoribbon, build_reduced_device, infer_centersize
from utils.energies import hamiltonian, multi_LDOS

from tqdm.auto import tqdm
import sisl
import numpy as np

import os
from pathlib import Path

from itertools import product

Nk = 1
ENERGIES = np.linspace(-2, 2, num=50)

# directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# the base dir to save the results
BASE_DIR = SCRIPT_DIR # QTM/results

def main(LWC_combinations):
    for L, W, C in tqdm(LWC_combinations, desc="LWC combinations", total=len(LWC_combinations)):
        folder = BASE_DIR / f"L{L}_W{W}_C{C}"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"saving data to folder : {folder}")
        electrode = build_electrode(width=W, length=L)
        ribbon = build_nanoribbon(electrode=electrode, center_size=C)
        device, lr_idx = build_reduced_device(ribbon, electrode, repeat=3)
        electrode.write(folder / "electrode.xyz")
        ribbon.write(folder / "ribbon.xyz")
        device.write(folder / "device.xyz")
        
        all_LDOS = multi_LDOS(device=device,
                              electrode=electrode,
                              lr_indices=lr_idx,
                              energies=ENERGIES,
                              Nk=Nk, eta=1e-5, form="csc"
                              )
        np.savez(folder / "calculations",
                 energies=ENERGIES,
                 ldos=all_LDOS,
                 lr_idx=lr_idx)
        
        pass



if __name__ == "__main__":
    Llist = range(1,3)
    ilist = range(4,8)
    all_inputs = product(Llist, ilist)
    unique_combinations = set(infer_centersize(l, i) for l, i in all_inputs)
    # print(unique_combinations)
    main(unique_combinations)