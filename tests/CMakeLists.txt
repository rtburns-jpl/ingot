set(tests
    two_body_1rev.cu
    simple_harmonic_oscillator.cu
    )

foreach(file ${tests})
    get_filename_component(namewe ${file} NAME_WE)
    set(target ${namewe})
    add_executable(${target} ${file})
    target_link_libraries(${target} PRIVATE ingot GTest::GTest GTest::Main)

    add_test(NAME test.${target} COMMAND ${target})
endforeach()
