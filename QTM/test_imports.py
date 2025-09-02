import sisl
import numpy
import scipy
import seaborn
import matplotlib
def main():
    print("Hello from qtm!")
    print("SISL version:", sisl.__version__)
    print("NumPy version:", numpy.__version__)
    print("SciPy version:", scipy.__version__)
    print("Seaborn version:", seaborn.__version__)
    print("Matplotlib version:", matplotlib.__version__)


if __name__ == "__main__":
    main()
