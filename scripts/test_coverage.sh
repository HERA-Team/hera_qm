#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

python setup.py install

cd hera_qm/tests
nosetests --with-coverage --cover-erase --cover-package=hera_qm --cover-html "$@"
