import sisl as si
import numpy as np
import ase
import utils 
import scipy
import matplotlib
import plotly

def main():
    
    all_pkgs = [si, np, ase, utils, scipy, matplotlib, plotly]
    print("Hello from qtm!\n Packages installed:")
    for pkg in all_pkgs:
        print(f"  {pkg.__name__} : {pkg.__version__}")
        

if __name__ == "__main__":
    main()