#!/bin/bash

# Install necessary packages using conda and mamba
conda install conda-forge:mamba
mamba install -c pytorch -c nvidia pytorch pytorch-cuda=12.1
mamba install -c conda-forge openmm-torch openmmtools
mamba install torch-scatter

# Ask the user to provide the root folder of CSnet
read -p "Please provide the root folder of GEqTrain: " geqtrain_root

# Save the current directory
current_dir=$(pwd)

# Check if the provided path is the basename "CSnet"
if [ "$(basename "$geqtrain_root")" == "GEqTrain" ]; then
    cd "$geqtrain_root"
else
    # Clone the CSnet repository to the provided path
    git clone https://github.com/limresgrp/GEqTrain.git "$geqtrain_root/GEqTrain"
    cd "$geqtrain_root/GEqTrain"
fi

# Install CSnet
pip install -e .

# Go back to the previous directory
cd "$current_dir"

# Ask the user to provide the root folder of CSnet
read -p "Please provide the root folder of CSnet: " csnet_root

# Check if the provided path is the basename "CSnet"
if [ "$(basename "$csnet_root")" == "CSnet" ]; then
    cd "$csnet_root"
else
    # Clone the CSnet repository to the provided path
    git clone https://github.com/Daniangio/CSnet.git "$csnet_root/CSnet"
    cd "$csnet_root/CSnet"
fi

# Install CSnet
pip install -e .

# Go back to the previous directory
cd "$current_dir"