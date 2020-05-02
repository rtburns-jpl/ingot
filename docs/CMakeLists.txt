find_package(Doxygen REQUIRED)
doxygen_add_docs(
    doxygen ${PROJECT_SOURCE_DIR}/ingot COMMENT "Build documentation"
    )
