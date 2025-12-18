from pathlib import Path
import numpy as np
import sisl
from utils.loader import load_datastructure, path_finder, parse_lwc
from utils.structure import find_nearest_atoms
from utils.helpers import in_notebook
from sisl.viz.processors.math import normalize

from warnings import warn

from tqdm.auto import tqdm
if in_notebook():
	TQDM_KWARGS = {"leave": True}
else:
	TQDM_KWARGS = {"leave": False}

BASE_DIR = path_finder("QTM")
RESULTS_DIR = BASE_DIR / "results"
LIST_OF_STRUCTURES = sorted(RESULTS_DIR.glob("L*_W*_C*"))

LIST_LWC = [parse_lwc(folder) for folder in LIST_OF_STRUCTURES] 

def find_bg(DATA, site, atol=0.01, kidx=None):
	data = dict(DATA)
	# tqdm.write(str(data.keys()))
	Nk, NE, Nsites = DATA["ldos"].shape
	if Nsites < site:
		warn(
      f"""Chosen site index exceeds number of sites in LDOS data. Chose {site} but only {Nsites} sites available.
      This is likely due to argument 'site' being chosen for center-most atom of device,
      while LDOS data only contains center region atoms (i.e., device with electrodes removed).
      Therefore, adjusting site index accordingly."""
       )
		site = 0
	if kidx == None:
		if Nk == 1:
			kidx = 0 # length is 1 	
		else: 
			raise ValueError(f"number of k-points for calculation : {Nk}, but chosen kidx : {kidx}")
	else: pass # using chosen kidx 
	for ldos_str in DATA.keys():
		if ("ldos" in ldos_str): # only consider ldos with modulation
			ldos = data[ldos_str]
			modulation = ldos_str.split("_")[1] if "_" in ldos_str else "C0"
			norm_ldos = normalize(ldos[kidx,:,site])
			energies = data["energies"]
			idxs = np.argwhere(norm_ldos <= atol) # find indecies for ldos ~ 0 
			bgmin = min(energies[idxs]) # find min energy level for bg
			bgmax = max(energies[idxs]) # find max energy level for bg
			data[f"bgmin_{modulation}"] = bgmin
			data[f"bgmax_{modulation}"] = bgmax
			data[f"bg_{modulation}"] = abs(bgmin - bgmax)
		else: 
			if not any([ldos_str.startswith(exclude_key) for exclude_key in ["energies", "lr_idx", "R", "bg"]]):
				print(f"{ldos_str} not found in DATA keys")
		
	return data
		
def get_center_region(device, lr_idx):
	center_region = device.remove(lr_idx)
	return center_region
def main():
	log_path: Path = RESULTS_DIR / "bg_log.txt"
	with open(log_path, "w") as logfile: # clar previous log file
		logfile.write("-- New Bandgap Calculation Run ---\n")
	for LWC in tqdm(LIST_LWC, desc="LWC structures", total=len(LIST_LWC), **TQDM_KWARGS):
		lwc_dir = RESULTS_DIR / "L{:}_W{:02}_C{:02}".format(*LWC)
		*_, device, DATA = load_datastructure(LWC)
		lr_idx = DATA["lr_idx"]
		center_region = get_center_region(device, lr_idx)
		xyz = center_region.xyz
		dist = np.linalg.norm(xyz[:, None,:] - xyz[:, :, None], axis=(1,2))
		site = find_nearest_atoms(xyz, device.center(), neighbours=1)
		DATA = find_bg(DATA, site, atol=0.01)
		outer_R = sorted(dist)[-1]/2 # find largest distance by : sorting in ascending order and taking last element
		DATA["R"] = outer_R
		realitive_path = (lwc_dir / "calculations.npz").relative_to(BASE_DIR.parent)
		
		np.savez(lwc_dir / "calculations.npz", 
			**DATA)
		with open(log_path, "a") as logfile:
			logfile.write(f"save bg calc to {realitive_path}\n")

	print("All calculations done!")


if __name__ == "__main__":
    
	main()
	with open(RESULTS_DIR / "bg_log.txt", "r") as logfile:
		print(logfile.read())
