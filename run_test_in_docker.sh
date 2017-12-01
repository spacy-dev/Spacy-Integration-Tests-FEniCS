#!/bin/bash

FENICS_SHARED="${FENICS_HOME}/shared"
DEPS="${FENICS_HOME}/deps"
INCLUDE_DIR=$HOME/include
LIB_DIR=$HOME/lib
TEST_DIR=$HOME/test

# Install gtest
mkdir ${DEPS}
cd ${DEPS}
git clone https://github.com/google/googletest.git
cd googletest && mkdir -p build && cd build && cmake .. && make -j2
sudo cp -r ../googletest/include/gtest $INCLUDE_DIR
sudo cp googlemock/gtest/lib*.a $LIB_DIR

# Install FunG
cd ${DEPS}
git clone https://github.com/lubkoll/FunG.git
sudo cp -r FunG/fung $INCLUDE_DIR

# Install Spacy
cd ${DEPS}
git clone https://github.com/spacy-dev/Spacy.git
cd Spacy && mkdir -p build && cd build && cmake .. -DDolfin=ON -DCMAKE_INSTALL_PREFIX=$HOME && make -j2 && sudo make install

# Run tests
mkdir $TEST_DIR
cp -r $FENICS_SHARED/* $TEST_DIR

cd ${TEST_DIR}/FEniCS
ffc -l dolfin LinearHeat.ufl
ffc -l dolfin L2Functional.ufl

cd ${TEST_DIR}
mkdir -p build && cd build
cmake .. -DDolfin=ON -DCMAKE_CXX_STANDARD=14
make -j2 && ctest

