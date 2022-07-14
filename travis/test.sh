#!/bin/bash
set -ev

if [ "$DISTRIBUTION" = "python" ]; then
    cd horizon/tests
    nosetests3 .
fi