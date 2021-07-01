cmake_minimum_required(VERSION 3.14...3.20)

project(ingot LANGUAGES CXX CUDA)

find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
endif()

if(NOT DEFINED CUDAToolkit_VERSION)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CUDAToolkit_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
    endif()
endif()

# CUDA's builtin Thrust is only compatible from
# toolkit version 10.1 (i.e. Thrust 1.9.4) onward
if(CUDAToolkit_VERSION VERSION_LESS 10.1)
    include(FetchContent)
    FetchContent_Declare(thrust
        GIT_REPOSITORY https://github.com/thrust/thrust.git
        GIT_SHALLOW ON
        GIT_TAG 1.9.6
        )
    if(NOT thrust_POPULATED)
        FetchContent_Populate(thrust)
    endif()
    # Include BEFORE to override system thrust
    include_directories(BEFORE SYSTEM ${thrust_SOURCE_DIR})
endif()

# Do CUDA compiler-specific prep work
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    find_package(Eigen3 REQUIRED)
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")

    # Eigen's latest stable release (3.3.7) has issues with Clang CUDA
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_SHALLOW ON
        )
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    add_library(Eigen3::Eigen IMPORTED INTERFACE)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

    # Clang CUDA has issues with GNU extensions
    set(CMAKE_CUDA_EXTENSIONS OFF)

else()
    message(WARNING "Unknown CUDA compiler id '${CMAKE_CUDA_COMPILER_ID}'")
endif()

include(GNUInstallDirs)

add_library(ingot INTERFACE)
target_include_directories(
    ingot INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
target_link_libraries(ingot INTERFACE Eigen3::Eigen)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_compile_features(ingot INTERFACE cuda_std_14)
else()
    set(CMAKE_CUDA_STANDARD 14)
endif()
target_compile_features(ingot INTERFACE cxx_std_14)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(ingot INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ingot INTERFACE
        -Wno-c++11-narrowing) # TODO fix these
endif()

option(BUILD_EXAMPLES "Build API usage examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

option(BUILD_BINDINGS "Build other language bindings" ON)
if(BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

option(ENABLE_TESTING "Enable unit tests" ON)
if(ENABLE_TESTING)
    find_package(GTest REQUIRED)

    enable_testing()
    add_subdirectory(tests)
endif()
