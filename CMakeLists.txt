cmake_minimum_required(VERSION 3.1)

option(Coverage "Coverage" OFF)

project(Spacy-Integration-Tests-FEniCS)

# Set path to cmake scripts
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

# Do not allow in-source builds
include(ConfigSafeGuards)

if(Coverage)
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-coverage)
endif()

include_directories(.)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

aux_source_directory(Mock MOCK_SRC_LIST)
add_library(mocks ${MOCK_SRC_LIST})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(DOLFIN CONFIG REQUIRED)
include(${DOLFIN_USE_FILE})
include_directories(${DOLFIN_INCLUDE_DIRS})
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})
add_definitions(${DOLFIN_CXX_DEFINITIONS})
aux_source_directory(FEniCS SRC_LIST)
find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} NO_MODULE QUIET)

find_package(Spacy CONFIG REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

include(CTest)
enable_testing()
foreach(TEST ${SRC_LIST})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  get_filename_component(TEST_DIR ${TEST} DIRECTORY)
  string(REGEX REPLACE "/" "_" TEST_DIR ${TEST_DIR})
  set(TEST_UNIQUE_NAME "${TEST_DIR}_${TEST_NAME}")
  add_executable(${TEST_UNIQUE_NAME} ${TEST})
  target_link_libraries(${TEST_UNIQUE_NAME} mocks ${DOLFIN_LIBRARIES} Spacy::Spacy GTest::GTest GTest::Main Threads::Threads)
add_test(${TEST_UNIQUE_NAME} ${PROJECT_BINARY_DIR}/${TEST_UNIQUE_NAME})
endforeach()
