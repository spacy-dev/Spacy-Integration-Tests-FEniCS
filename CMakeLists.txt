cmake_minimum_required(VERSION 3.1)

option(Coverage "Coverage" OFF)
option(Kaskade "Kaskade" OFF)
option(Dolfin "Dolfin" OFF)
option(Eigen "Eigen" OFF)
option(dealII "dealII" OFF)

if(dealII)
  find_package(deal.II 8.4 QUIET
      HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
      )
  DEAL_II_INITIALIZE_CACHED_VARIABLES()
endif()

project(Spacy-Integration-Tests)

# Set path to cmake scripts
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

# Do not allow in-source builds
include(ConfigSafeGuards)

if(Coverage)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage")
endif()

include_directories(.)

find_package(Spacy CONFIG REQUIRED)
include_directories(${Spacy_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Spacy_CXX_FLAGS}")

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


aux_source_directory(Mock MOCK_SRC_LIST)
add_library(mocks ${MOCK_SRC_LIST})

if(Eigen)
  find_package(Eigen3 REQUIRED)
  include_directories(${EIGEN3_INCLUDE_DIR})
  aux_source_directory(Eigen SRC_LIST)
endif()


if(Dolfin) 
  find_package(DOLFIN CONFIG REQUIRED)
  include_directories(${DOLFIN_INCLUDE_DIRS})
  include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})
  add_definitions(${DOLFIN_CXX_DEFINITIONS})
  aux_source_directory(FEniCS SRC_LIST)
  find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} NO_MODULE QUIET)
endif()

if(dealII)
    aux_source_directory(dealII SRC_LIST)
endif()

if(Kaskade)
  include_directories(/home/lars/Libraries/Kaskade7.3
                      /opt/acml5.3.1/gfortran64/include
                      /home/lars/Libraries/software/gcc-5.4.0/dune-2.4.0/include
                      /home/lars/Libraries/software/gcc-5.4.0/umfpack-5.4.0/include
                      /home/lars/Libraries/software/gcc-5.4.0/dune-2.4.0/include/ug)
  link_directories(/home/lars/Libraries/Kaskade7.3/libs
                   /opt/acml5.3.1/gfortran64/lib
                   /home/lars/Libraries/software/gcc-5.4.0/umfpack-5.4.0/lib
                   /home/lars/Libraries/software/gcc-5.4.0/dune-2.4.0/lib)
  aux_source_directory(Kaskade SRC_LIST)
endif()


include(CTest)
enable_testing()
foreach(TEST ${SRC_LIST})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  get_filename_component(TEST_DIR ${TEST} DIRECTORY)
  string(REGEX REPLACE "/" "_" TEST_DIR ${TEST_DIR})
  set(TEST_UNIQUE_NAME "${TEST_DIR}_${TEST_NAME}")
  add_executable(${TEST_UNIQUE_NAME} ${TEST})
  target_link_libraries(${TEST_UNIQUE_NAME} mocks ${Spacy_LIBRARIES} ${GTEST_LIBRARIES} pthread)
  if(Kaskade)
    target_link_libraries(${TEST_UNIQUE_NAME} kaskade boost_system boost_timer boost_thread dunegrid dunecommon dunegeometry ugS3 ugS2 ugL3 ugS2 devS devX umfpack amd acml)
  endif()
  if(Dolfin)
    target_link_libraries(${TEST_UNIQUE_NAME} ${DOLFIN_LIBRARIES})
  endif()
  if(dealII)
    DEAL_II_SETUP_TARGET(${TEST_UNIQUE_NAME})
  endif()
add_test(${TEST_UNIQUE_NAME} ${PROJECT_BINARY_DIR}/${TEST_UNIQUE_NAME})
endforeach()
