import sisl
import numpy
import scipy
import seaborn
import matplotlib
import ase
SHIFT = 20

libs = {
    'sisl': sisl,
    'numpy': numpy,
    'scipy': scipy,
    'seaborn': seaborn,
    'matplotlib': matplotlib,
    'ase': ase,
}
def main():
    print(f"Hello from qtm!")
    for lib_name, lib in libs.items():
        print(f"{lib_name.title() + ' version':>{SHIFT}}:", lib.__version__)






if __name__ == "__main__":
    main()
