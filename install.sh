#!/bin/bash

# Create a new conda environment with Python 3.7
conda create --name GraOmicCombine python=3.7

# activate env
conda activate GraOmicCombine

# install rdkit with conda
conda install rdkit

# install requirements with pip
pip install -r requirements.txt
