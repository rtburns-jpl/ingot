cmake_minimum_required(VERSION 3.17)

project(eigen_ode LANGUAGES CUDA)

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cuda_std_14)

find_package(Eigen3 REQUIRED)

add_executable(main main.cu)
target_link_libraries(main PRIVATE Eigen3::Eigen project_options)

target_include_directories(main PRIVATE thrust)
