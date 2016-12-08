#!/bin/bash

FENICS_SHARED="${FENICS_HOME}/shared"
DEPS="${FENICS_SHARED}/deps"

# Install gtest
git clone https://github.com/google/googletest.git
cd googletest && mkdir build && cd build && cmake .. && make -j2
sudo cp -r ../googletest/include/gtest /usr/local/include/
sudo cp googlemock/gtest/lib*.a /usr/local/lib

# Install FunG
cd ${DEPS}
git clone https://github.com/lubkoll/FunG.git
sudo cp -r FunG/fung /usr/local/include

# Install Spacy
cd ${DEPS}
git clone https://github.com/spacy-dev/Spacy.git
cd Spacy && mkdir build && cd build && cmake .. -DDolfin=ON && make -j2 && sudo make install

# Run tests
cd ${FENICS_SHARED}/FEniCS
ffc -l dolfin LinearHeat.ufl
ffc -l dolfin L2Functional.ufl

cd ${FENICS_SHARED}
mkdir build && cd build
cmake .. -DDolfin=ON
make -j2 && ctest

