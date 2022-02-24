#!/bin/bash

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
printf '\nyes\n\nyes\n' | bash Mambaforge-$(uname)-$(uname -m).sh

export PATH /home/daniele/mambaforge/bin:$PATH

yes Y | mamba env create -f environment.yml