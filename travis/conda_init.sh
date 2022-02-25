#!/usr/bin/env bash

set -ev

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
printf '\nyes\n\nyes\n' | bash Mambaforge-$(uname)-$(uname -m).sh



echo "before sourcing: $PATH"
# re-activate shell
export PATH=$PWD/mambaforge/bin:$PATH
source ~/.bashrc

echo "after sourcing: $PATH"
# create environment for conda
yes Y | mamba env create -f environment.yml