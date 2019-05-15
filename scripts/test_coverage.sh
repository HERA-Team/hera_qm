#! /bin/bash
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

python setup.py install

cd hera_qm/tests
python -m pytest --cov=hera_qm --cov-config=../../.coveragerc\
       --cov-report term --cov-report html:cover \
       "$@"
