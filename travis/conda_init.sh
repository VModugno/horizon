#!/bin/bash
# -i stands for interactive mode: required to make effective source ~/.bashrc
set -v

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
printf '\nyes\n\nyes\n' | bash Mambaforge-$(uname)-$(uname -m).sh


# re-activate shell
echo "PATH=$PWD/mambaforge/bin:$PATH" >> ~/.bashrc
# export PATH=$PWD/mambaforge/bin:$PATH
# source ~/.bashrc 

yes Y | mamba env create -f environment.yml

# create environment for conda
