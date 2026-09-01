cmake_minimum_required(VERSION 3.16)

project(v5_host C)

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

add_executable(
    v5_f26_host
    test_v5_f26_model.c
    ../../Modules/Src/v5_model.c
    ../../Modules/Src/v5_runtime.c
    ../../Modules/Src/v5_stim_fsm.c
)

target_include_directories(
    v5_f26_host
    PRIVATE
    ../../Modules/Inc
)

add_executable(
    v5_runtime_parity
    test_v5_runtime_parity.c
    ../../Modules/Src/v5_model.c
    ../../Modules/Src/v5_runtime.c
)

add_executable(
    v5_candidate_parity
    test_v5_candidate_parity.c
    ../../Modules/Src/v5_candidate.c
)

add_executable(
    v5_feature_parity
    test_v5_feature_parity.c
    ../../Modules/Src/v5_features.c
)

add_executable(
    v5_full_replay
    test_v5_full_replay.c
    ../../Modules/Src/v5_candidate.c
    ../../Modules/Src/v5_features.c
    ../../Modules/Src/v5_model.c
    ../../Modules/Src/v5_runtime.c
    ../../Modules/Src/v5_subject_config.c
)

add_executable(
    v5_safety
    test_v5_safety.c
    ../../Core/Src/v5_integration_example.c
    ../../Modules/Src/v5_candidate.c
    ../../Modules/Src/v5_features.c
    ../../Modules/Src/v5_model.c
    ../../Modules/Src/v5_runtime.c
    ../../Modules/Src/v5_stim_fsm.c
    ../../Modules/Src/v5_subject_config.c
)

target_include_directories(
    v5_runtime_parity
    PRIVATE
    ../../Modules/Inc
)

target_include_directories(
    v5_candidate_parity
    PRIVATE
    ../../Modules/Inc
)

target_include_directories(
    v5_feature_parity
    PRIVATE
    ../../Modules/Inc
)

target_include_directories(
    v5_full_replay
    PRIVATE
    ../../Modules/Inc
)

target_include_directories(
    v5_safety
    PRIVATE
    ../../Modules/Inc
)

if(NOT MSVC)
    target_link_libraries(
        v5_f37_host
        m
    )
    target_link_libraries(
        v5_f26_host
        m
    )
    target_link_libraries(
        v5_runtime_parity
        m
    )
    target_link_libraries(
        v5_candidate_parity
        m
    )
    target_link_libraries(
        v5_feature_parity
        m
    )
    target_link_libraries(
        v5_full_replay
        m
    )
    target_link_libraries(
        v5_safety
        m
    )
endif()
