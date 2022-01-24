cmake_minimum_required(VERSION 3.21)
project(PETScTutorial)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER ${PETSC_DIR}/bin/mpicxx)
set(CMAKE_C_COMPILER ${PETSC_DIR}/bin/mpicc)
# message(${CMAKE_CXX_COMPILER})
# message(${CMAKE_C_COMPILER})

# Add path to petsc module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# NOTE: NEED TO HAVE THE FOLLOWING VARIABLES
#  PETSC_DIR - directory in which PETSc resides
#  PETSC_ARCH - build architecture
find_package(PETSc REQUIRED)

add_executable(PETScTutorial main.cpp)

# Include directories
target_include_directories(PETScTutorial PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(PETScTutorial PUBLIC ${PETSC_INCLUDES})

# Linking libraries
target_link_libraries(     PETScTutorial PUBLIC ${PETSC_LIBRARIES})
