# MMlib

MMlib is a library that implements different approaches to save and recover models. It was developed as part of my
master thesis ([link to thesis repo](https://github.com/slin96/master-thesis)).

The approach names in the thesis match the following implementations:
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

- For examples on how to use mmlib and the probing tool checkout the [examples](examples) directory.
    
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



