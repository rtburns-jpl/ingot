# These should all compile and run. Output is not checked for correctness.
set(examples
    simple_harmonic_oscillator.cu
    two_body.cu
    poincare.cu
    )

foreach(file ${examples})
    get_filename_component(namewe ${file} NAME_WE)
    set(namewe "example.${namewe}")
    set(target ${namewe})
    add_executable(${target} ${file})
    target_link_libraries(${target} PRIVATE ingot)
endforeach()
