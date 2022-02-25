echo "before sourcing: $PATH"
# re-activate shell
export PATH=$PWD/mambaforge/bin:$PATH
source ~/.bashrc

echo "after sourcing: $PATH"
# create environment for conda
yes Y | mamba env create -f environment.yml