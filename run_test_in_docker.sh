#!/bin/bash

FENICS_SHARED="${FENICS_HOME}/shared"
DEPS="${FENICS_HOME}/deps"
INCLUDE_DIR=$HOME/include
LIB_DIR=$HOME/lib
TEST_DIR=$HOME/test

mkdir $DEPS
mkdir $INCLUDE_DIR
mkdir $LIB_DIR
export PATH=$PATH:$INCLUDE_DIR:$LIB_DIR

# Install gtest
cd ${DEPS}
git clone https://github.com/google/googletest.git
cd googletest && mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME
cmake --build . && cmake --build . --target install

# Install FunG
cd ${DEPS}
git clone https://github.com/lubkoll/FunG.git
cp -r FunG/fung $INCLUDE_DIR

# Install Spacy
cd ${DEPS}
git clone https://github.com/spacy-dev/Spacy.git
cd Spacy && mkdir -p build && cd build && cmake .. -DDolfin=ON -DCMAKE_INSTALL_PREFIX=$HOME -DCMAKE_CXX_FLAGS=-I/usr/local/lib/python3.6/dist-packages/ffc/backends/ufc
cmake --build . && cmake --build . --target install

# Run tests
mkdir $TEST_DIR
cp -r $FENICS_SHARED/* $TEST_DIR

cd ${TEST_DIR}/FEniCS
ffc -l dolfin LinearHeat.ufl
ffc -l dolfin L2Functional.ufl

cd ${TEST_DIR}
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS=-I/usr/local/lib/python3.6/dist-packages/ffc/backends/ufc
make -j2 && ctest

