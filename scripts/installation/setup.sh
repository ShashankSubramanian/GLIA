#!/bin/bash

echo "Creating python3 virtual environment."
python3 -m venv ../../py_env
echo "  .. done"
unset PYTHONPATH
source ../../py_env/bin/activate


echo "installing python dependencies"
pip3 intall --upgrade pip
pip3 install -r requirements.txt
echo "done, exiting."
