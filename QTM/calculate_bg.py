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

def find_bg(DATA, site, atol=0.01, energy_window=1.0, kidx=None):
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
			
			near_fermi = np.abs(energies) <= energy_window
		
			is_gap = (norm_ldos <= atol) & near_fermi
			bgmin = energies[is_gap].min() if np.any(is_gap) else 0.0
			bgmax = energies[is_gap].max() if np.any(is_gap) else 0.0
			# if not np.any(is_gap):
			# 	tqdm.write(f"No bandgap found for modulation {modulation} at site {site} with atol={atol} and energy_window={energy_window} eV.")
			# 	bgmin = 0.0
			# 	bgmax = 0.0
			# 	continue
			# # indicies qualifying as bandgap
			# gap_indices = np.where(is_gap)[0]
			# # Find the index of the energy level closest to E=0
			# zero_idx = np.argmin(np.abs(energies))
			# if len(gap_indices) > 0:
       		# 	# 1. group into contiguous islands
			# 	islands = np.split(gap_indices, np.where(np.diff(gap_indices) > 1)[0] + 1)
			# 	# 3. Check if island contains zero_idx
			# 	best_island = None
			# 	min_dist = float("inf")
			# 	for isl in islands:
			# 		if isl.min() <= zero_idx <= isl.max():
			# 			best_island = isl
			# 			break # found the 'true' gap crossing E=0
			# 		dist = min(np.abs(isl - zero_idx))
			# 		if dist < min_dist:
			# 			min_dist = dist
			# 			best_island = isl
			# 	best_island = min(islands, key=lambda isl: np.abs(isl.mean() - zero_idx))
			# 	vbm_idx, cbm_idx = best_island.min(), best_island.max()
			# 	bgmin, bgmax = energies[vbm_idx], energies[cbm_idx] # find min energy level for bg
			# else:
			# 	bgmin, bgmax = 0.0, 0.0
   
   
			# 4. ROBUST STEP: Only consider the gap that actually crosses or touches E=0
			# if zero_idx in gap_indices:
			# 	# find contiguous gap region around E=0
			# 	start_gap = gap_indices[gap_indices <= zero_idx][-1]
			# 	end_gap = gap_indices[gap_indices >= zero_idx][0]

			# 	vbm_idx = gap_indices[np.where(gap_indices == start_gap)[0][0]]
			# 	while vbm_idx - 1 in gap_indices: vbm_idx -= 1

			# 	cbm_idx = gap_indices[np.where(gap_indices == end_gap)[0][0]]
			# 	while cbm_idx + 1 in gap_indices: cbm_idx += 1
			# else:
			# 	# If E=0 isn't in a gap, we take the absolute min/max of the detected region
			# 	vbm_idx = gap_indices.min()
			# 	cbm_idx = gap_indices.max()
			
   
			
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
		DATA = find_bg(DATA, site, atol=0.01, energy_window=0.5)

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
