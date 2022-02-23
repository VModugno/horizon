#!/bin/bash

# upload to pypi
python3 -m build --wheel;
twine upload -u __token__ -p $PYPI_TOKEN dist/*.whl;


# upload to docker
cd $HORIZON_DIR/docker
./upload.sh