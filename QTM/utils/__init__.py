__version__ = "0.2.2"

import sisl
import numpy as np
from ._wrappers import timeit
from .structure import build_electrode, build_nanoribbon, build_reduced_device, infer_centersize, find_nearest_atoms
from .energies import hamiltonian, lr_energies, add_lr_energies, LDOS
from .loader import path_finder, load_datastructure
from .helpers import in_notebook
# from .energies import multi_LDOS_parallel as multi_ldos

# from .plots import plot_with_center


__all__ = ["timeit",
           "build_electrode", "build_nanoribbon", "build_reduced_device", 
            "infer_centersize", "find_nearest_atoms",
           "hamiltonian", "lr_energies", "add_lr_energies",
           "LDOS", "multi_ldos_parallel",
           "path_finder", "load_datastructure",
           "in_notebook",
       #     "plot_with_center",
           "__version__",
           
    ]



def pretty_print_columns(A, decimals=2, zero_repr="0"):
    """
    Print a 2D NumPy array with per-column alignment.
    Handles complex numbers, negatives, and zeros gracefully.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array")

    rows, cols = A.shape
    is_complex = np.iscomplexobj(A)

    # Build a string matrix (formatted values)
    str_matrix = np.empty(A.shape, dtype=object)
    for i in range(rows):
        for j in range(cols):
            val = A[i, j]
            if val == 0:
                s = zero_repr
            elif is_complex:
                s = f"{val.real:.{decimals}f}{val.imag:+.{decimals}f}j"
            else:
                s = f"{val:.{decimals}f}"
            str_matrix[i, j] = s

    # Compute per-column widths
    col_widths = [max(len(str_matrix[i, j]) for i in range(rows)) for j in range(cols)]

    # Print rows with proper per-column alignment
    for i in range(rows):
        row_str = " ".join(str_matrix[i, j].rjust(col_widths[j]) for j in range(cols))
        print(row_str)


