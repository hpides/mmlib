# mmlib

Add a description here!

## installation
To build the library you have two options.

### Option1: Docker
- **Requirements**: Docker installed
- **Build Library**
  - run the script `generate-archives-docker.sh`
  - it runs a docker container and builds the *mmlib* in it. 
  - the created `dist`directory is copied back to repository root
  - it contains the `.whl` file that can used to install the library with pip (see below) 
 

- to build the lib run: `generate-archives.sh`
- to install it run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl
  `