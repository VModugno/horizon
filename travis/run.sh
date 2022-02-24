#!/bin/sh

# run python or conda distribution
if [ "$DISTRIBUTION" = "python" ]; then
    echo "calling"
    source ./travis/python_run.sh;
fi

if [ "$DISTRIBUTION" = "conda" ]; then
    source ./travis/conda_run.sh;
fi
