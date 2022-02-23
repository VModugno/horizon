if [ "$DISTRIBUTION" = "python" ]; then
    ./travis/python_compile.sh;
fi

if [ "$DISTRIBUTION" = "conda" ]; then
    ./travis/conda_compile.sh;
fi
