find_package(Python 3.6 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)

set(INGOT_PY_EXT ingot_ext)
Python_add_library(${INGOT_PY_EXT} ext.cu)
target_compile_definitions(${INGOT_PY_EXT} PRIVATE
    -DPY_EXT_NAME=${INGOT_PY_EXT})
target_link_libraries(${INGOT_PY_EXT} PRIVATE
    ingot
    )

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # TODO also use -fvisibility=hidden
    target_include_directories(${INGOT_PY_EXT} PRIVATE
        ${pybind11_INCLUDE_DIRS}
        )
else()
    target_link_libraries(${INGOT_PY_EXT} PRIVATE
        pybind11::module
        )
endif()

if(NOT DEFINED INGOT_PYPKG_DIR)
    set(INGOT_PYPKG_DIR "packages" CACHE STRING
        "Python package dir, relative to install prefix"
        )
endif()

install(TARGETS ${INGOT_PY_EXT}
        DESTINATION ${INGOT_PYPKG_DIR}/ingot
        )
install(FILES __init__.py
        DESTINATION ${INGOT_PYPKG_DIR}/ingot
        )
