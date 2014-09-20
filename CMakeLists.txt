cmake_minimum_required(VERSION 2.8)

project(kondo)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# -- COMPILE OPTIONS -------------------------------------------

# MPI
set(WITH_MPI no CACHE BOOL "Build with MPI support")

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

# -- C++11 -------------------------------------------

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

# Armadillo
find_package(Armadillo 4.400.2 REQUIRED CONFIG
             PATHS "/usr/local" "$ENV{ARMA_ROOT}" NO_SYSTEM_ENVIRONMENT_PATH)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES})

# Thread Building Blocks
find_package(TBB)
if (TBB_FOUND)
    include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${TBB_LIBRARIES})
    add_definitions(-DWITH_TBB)
endif()

# FastKPM
find_package(FastKPM REQUIRED)
include_directories(SYSTEM ${FASTKPM_INCLUDE_DIRS})
set(LIBS ${LIBS} ${FASTKPM_LIBRARIES})


# -- BUILD -------------------------------------------

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
message("WITH_MPI: ${WITH_MPI}")
message("")
