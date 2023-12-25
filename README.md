# GAMR
Learning geometrically aware molecular representations

[Google doc](https://docs.google.com/document/d/1QwHjj8ZEuONPoNCUCVoGG_hF3o2PNJSgcTQkLIAGDFc/edit)

## Files:

* ``README.md``: me!
* ``.gitignore``: standard .gitignore file
* ``environment.yml``: conda environment, install with  ``conda env create -f environment.yml``
* ``tests.py``: unit tests; run with ``python tests.py``
* ``atomic_symbols.json``: hard coded json file where keys are atomic numbers and values are atomic symbols

### ``data_pipeline``:
Pipeline of blackbox that computes ground truth distance field for 3D molecular conformers.
* ``compute_dist.py``: contains most of the functions for computing ground truth distance field; the main file in this directory
* ``strings``: plaintext file of ~250k NCI/DTP database molecule SMILES; for checking which elements show up in drugs
* ``get_unique_atoms.ipynb``: short notebook for computing how many unique atoms are in NCI/DTP database molecules
* ``benchmarking.py``: benchmarking script for conformer generation and distance field calculation, writes results to ``benchmarking.txt`` when run


