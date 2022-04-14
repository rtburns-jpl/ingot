add_executable(ingot_tests
    doctest_main.cpp
    #dense_output.cu
    two_body_1rev.cu
    simple_harmonic_oscillator.cu
    simple_harmonic_oscillator_single.cu
    )
target_link_libraries(ingot_tests PRIVATE ingot doctest::doctest)
doctest_discover_tests(ingot_tests)
target_compile_definitions(ingot_tests PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
