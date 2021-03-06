cmake_minimum_required(VERSION 2.8)

project(kondo)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# -- COMPILE OPTIONS -------------------------------------------

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

# C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Aborting.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Mac fixes
if (APPLE)
  cmake_policy(SET CMP0042 NEW) # @rpath
  set(LIBS ${LIBS} "-framework Accelerate")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()


# -- DEPENDENCIES -------------------------------------------

# Thread Building Blocks
find_package(TBB)
if (TBB_FOUND)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${TBB_LIBRARIES})
    add_definitions(-DWITH_TBB)
endif()

# MPI
find_package(MPI)
if (WITH_MPI)
  include_directories(${MPI_INCLUDE_PATH})
  set(LIBS ${LIBS} ${MPI_LIBRARIES})
  set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
  add_definitions(-DWITH_MPI)
endif()

# Armadillo
find_package(Armadillo REQUIRED CONFIG
             PATHS "/usr/local" "$ENV{ARMA_ROOT}" NO_SYSTEM_ENVIRONMENT_PATH)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES})

# FastKPM
find_package(fastkpm REQUIRED)
include_directories(SYSTEM ${FASTKPM_INCLUDE_DIRS})
set(LIBS ${LIBS} ${FASTKPM_LIBRARIES})

# Boost
find_package(Boost 1.55.0 REQUIRED COMPONENTS filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


# -- BUILD -------------------------------------------

include_directories(src)
add_subdirectory(src)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("C++ compiler        : ${CMAKE_CXX_COMPILER}")
message("C++ compiler flags  : ${CMAKE_CXX_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("WITH_MPI: ${MPI_FOUND}")
message("")
