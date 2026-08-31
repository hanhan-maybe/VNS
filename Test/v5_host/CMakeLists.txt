cmake_minimum_required(VERSION 3.16)

project(v5_f37_host C)

set(CMAKE_C_STANDARD 11)

add_executable(
    v5_f37_host

    test_v5_f37_model.c

    ../../Modules/Src/v5_model.c
    ../../Modules/Src/v5_runtime.c
    ../../Modules/Src/v5_stim_fsm.c
)

target_include_directories(
    v5_f37_host
    PRIVATE
    ../../Modules/Inc
)

if(NOT MSVC)
    target_link_libraries(
        v5_f37_host
        m
    )
endif()
