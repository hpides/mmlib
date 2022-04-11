<h1 align="center">Efficiently Managing Deep Learning Models in a Distributed Environment </h1>
<p align="center">This repository contains the code to our <a href="https://openproceedings.org/2022/conf/edbt/paper-60.pdf"> EDBT '22 paper<a/>.<p/>


# MMlib

MMlib is a library that implements different approaches to save and recover models.

Approach names:
- baseline approach 
    - implemented by the `BaselineSaveService`
- parameter update approach 
    - implemented by `WeightUpdateSaveService` (set `improved_version=False`)
- improved parameter update approach 
    - implemented by `WeightUpdateSaveService` (set `improved_version=True`)
- provenance approach
    - implemented by `ProvenanceSaveService`
    
Next to the approaches to save and recover models we also implemented a **probing tool**
- the corresponding code is in `probe.py`

## Examples

- For examples on how to use MMlib and the probing tool checkout the [examples](examples) directory.
    
## Installation

### Option 1: Docker

- **Requirements**: Docker installed
- **Build Library**
    - clone this repo
    - run the script `generate-archives-docker.sh`
        - it runs a docker container and builds the *mmlib* in it
        - the created `dist` directory is copied back to repository root
        - it contains the `.whl` file that can be used to install the library with pip (see below)
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`

### Option 2: Local Build

- **Requirements**: Python 3.8 and Python `venv`
- **Build Library**
    - run the script `generate-archives.sh`
        - it creates a virtual environment, activates it, and installs all requirements
        - afterward it builds the library, and a `dist` directory containing the `.whl` file is created
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`


### Cite Our Work
If you use MMlib or insights from the paper, please cite us.

```bibtex
@inproceedings{strassenburg_2022_mmlib,
  author    = {Nils Strassenburg and Ilin Tolovski and Tilmann Rabl},
  title     = {Efficiently Managing Deep Learning Models in a Distributed Environment},
  booktitle = {Proceedings of the 25th International Conference on Extending Database Technology (EDBT 2022) Edinburgh, UK, March 29 - April 1},
  pages     = {234--246},
  publisher = {OpenProceedings.org},
  year      = {2022},
  doi       = {10.48786/edbt.2022.12}
}
```
