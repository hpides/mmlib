#!/bin/bash

# operate in the current dir
cd "$(dirname "$0")"

# install requirements
python3 -m pip install --upgrade setuptools wheel
python3 -m pip install -r requirements-tests.txt

python3 setup.py sdist bdist_wheel