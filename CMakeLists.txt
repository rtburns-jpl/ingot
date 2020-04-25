cmake_minimum_required(VERSION 3.17)

project(ingot LANGUAGES CUDA)

find_package(Eigen3 REQUIRED)

include(GNUInstallDirs)

add_library(ingot INTERFACE)
target_include_directories(
    ingot INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
target_link_libraries(ingot INTERFACE Eigen3::Eigen)
target_compile_features(ingot INTERFACE cuda_std_14)

option(BUILD_EXAMPLES "Build API usage examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(ENABLE_TESTING "Enable unit tests" ON)
if(ENABLE_TESTING)
    enable_language(CXX)
    find_package(GTest REQUIRED)

    enable_testing()
    add_subdirectory(tests)
endif()

add_executable(main main.cu)
target_link_libraries(main PRIVATE ingot)
