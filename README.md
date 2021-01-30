# mmlib

- A library for model management and related tasks.

## Installation

### Option 1: Docker

- **Requirements**: Docker installed
- **Build Library**
    - clone this repo
    - run the script `generate-archives-docker.sh`
      - it runs a docker container and builds the *mmlib* in it.
      - the created `dist` directory is copied back to repository root
      - it contains the `.whl` file that can be used to install the library with pip (see below)
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`

### Option 2: Local Build

- **Requirements**: Python 3.8
- **Build Library**
    - run the script `generate-archives.sh`
      - it creates a virtual environment, activates it, and installs all requirements
      - afterward it builds the library, and a `dist` directory containing the `.whl` file is created
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`

## Examples

- For examples on how to use mmlib checkout the [examples](./examples) directory.



