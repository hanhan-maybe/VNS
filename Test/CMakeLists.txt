
# ====================================================================
#  VNS ?????? PC ????????(CMake)
# ====================================================================
cmake_minimum_required(VERSION 3.16)
project(vns_unit_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ?????? ?????????????add_library(test_support OBJECT test_framework.c)

# ????????set(MODULE_SRC
    ${CMAKE_SOURCE_DIR}/../Modules/Src/ring_buffer.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/signal_processing.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/feature_extraction.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/classifier.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/frame_protocol.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/stim_trigger.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/logger.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/vns_config.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/system_manager.c
    ${CMAKE_SOURCE_DIR}/../Modules/Src/telemetry.c
)
add_library(modules STATIC ${MODULE_SRC})
target_include_directories(modules PUBLIC
    ${CMAKE_SOURCE_DIR}/../Modules/Inc
)

# ???????????add_executable(run_tests
    test_main.c
    test_ring_buffer.c
    test_signal_processing.c
    test_feature_extraction.c
    test_classifier.c
    test_frame_protocol.c
)

target_link_libraries(run_tests PRIVATE modules test_support)
target_include_directories(run_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/../Modules/Inc
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ?????? math ??target_link_libraries(run_tests PRIVATE m)

# ??????
add_custom_target(test
    COMMAND run_tests
    DEPENDS run_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ???????? CTest ???
enable_testing()
add_test(NAME vns_unit_tests COMMAND run_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

