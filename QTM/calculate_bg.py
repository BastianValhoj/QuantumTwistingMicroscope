import numpy as np
import sisl
from utils.loader import load_datastructure, path_finder
from utils.structure import find_nearest_atoms
from sisl.viz.processors.math import normalize


BASE_DIR = path_finder("QTM")
RESULTS_DIR = BASE_DIR / "results"
LIST_OF_STRUCTURES = sorted(RESULTS_DIR.glob("L*_W*_C*"))
def parse_lwc(folder):
	return tuple(int(part[1:]) for part in folder.name.split("_"))
LIST_LWC = [parse_lwc(folder) for folder in LIST_OF_STRUCTURES] 

def find_bg(DATA, site, atol=0.01, kidx=None):
	ldos = DATA["ldos"]
	Nk, NE, Nsites = ldos.shape
	if kidx == None:
		if Nk == 1:
			kidx = 0 # length is 1 	
		else: 
			raise ValueError(f"number of k-points for calculation : {Nk}, but chosen kidx : {kidx}")
	else: pass # using chosen kidx 
	
	norm_ldos = normalize(ldos[kidx,:,site])
	energies = DATA["energies"]
	idxs = np.argwhere(norm_ldos <= atol) # find indecies for ldos ~ 0 
	bgmin = min(energies[idxs]) # find min energy level for bg
	bgmax = max(energies[idxs]) # find max energy level for bg
	bg = abs(bgmin - bgmax)
	return bgmin, bgmax, bg
		

def main():
	for LWC in LIST_LWC:
		lwc_dir = RESULTS_DIR / "L{:}_W{:02}_C{:02}".format(*LWC)
		*_, device, DATA = load_datastructure(LWC)
		lr_idx = DATA["lr_idx"]
		center_region = device.sub(lr_idx)
		xyz = center_region.xyz
		dist = np.linalg.norm(xyz[:, None,:] - xyz[:, :, None], axis=(1,2))
		site = find_nearest_atoms(xyz, device.center(), neighbours=1)
		bgmin, bgmax, bg = find_bg(DATA, site, atol=0.01)
		outer_R = sorted(dist)[-1] # find largest distance by : sorting in ascending order and taking last element
		file_name = lwc_dir / "bandgap_calc"
		with open(RESULTS_DIR / "bg_log.txt", "a") as logfile:
			logfile.write(f"save bg calc to {file_name}\n")
			#print(f"save bg calc to {file_name}", file=logfile)
		bg_dict = {"min": bgmin, "max": bgmax, "bg": bg, "R": outer_R}
		np.savez(lwc_dir / "bandgap_calc", 
			**bg_dict)
		
	pass





if __name__ == "__main__":
	main()
	print("done!")
