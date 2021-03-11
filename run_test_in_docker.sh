#!/bin/bash

set -e

sudo apt update
sudo apt install python3 python3-pip python3-venv liblapacke-dev -y
python3 -m pip install --user pipx
python3 -m pipx ensurepath
export PATH="${PATH}:${HOME}/.local/bin"
pipx install conan
conan profile new default
conan profile update settings.compiler=gcc default
conan profile update settings.compiler.version=7 default
conan profile update settings.arch=x86_64 default
conan profile update settings.os=Linux default
conan profile update settings.compiler.libcxx=libstdc++11 default

sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3 99

FENICS_SHARED="${FENICS_HOME}/shared"
DEPS="${FENICS_HOME}/deps"
INCLUDE_DIR=$HOME/include
LIB_DIR=$HOME/lib
TEST_DIR=$HOME/test

mkdir $DEPS
mkdir $INCLUDE_DIR
mkdir $LIB_DIR
export PATH=$PATH:$INCLUDE_DIR:$LIB_DIR

# Install FunG
cd $DEPS
git clone https://github.com/lubkoll/FunG
cd FunG && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INCLUDE_DIR && cmake --build . --target install

# Install Spacy
cd ${DEPS}
git clone https://github.com/spacy-dev/Spacy.git
cd Spacy && mkdir -p build && cd build
conan install .. --build=missing
cmake .. -DDolfin=ON -DCMAKE_INSTALL_PREFIX=$HOME -DCMAKE_CXX_FLAGS=-I/usr/local/lib/python3.6/dist-packages/ffc/backends/ufc -DCMAKE_TOOLCHAIN_FILE=conan_paths.cmake
cmake --build . --target install

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

