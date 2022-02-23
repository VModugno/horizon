#!/bin/sh

if [ '$DISTRIBUTION' = python ]; then
    python3 horizon/tests/test_get_set.py
fi