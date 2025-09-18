from setuptools import setup, find_packages
import os

# Read the version from utils/__init__.py
version = {}
with open(os.path.join("utils", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)

setup(
    name="qtm-utils",
    version=version["__version__"],
    packages=find_packages(),
    install_requires=[
        "qutip",
        "numpy",
        "plotly",
        "matplotlib",
        "ipykernel",
        "ipympl",
        "nbformat",
    ],
)
