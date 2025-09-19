cmake_minimum_required(VERSION 3.16)
project(SentioTransformerStrategy VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Find required packages
find_package(Torch REQUIRED)
find_package(CURL QUIET)

# OpenMP is optional on macOS/AppleClang
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found; proceeding without OpenMP")
endif()

# yaml-cpp optional
find_package(yaml-cpp QUIET CONFIG)
if(TARGET yaml-cpp)
    set(HAVE_YAML TRUE)
else()
    set(HAVE_YAML FALSE)
    message(WARNING "yaml-cpp not found; tests or tools requiring YAML may fail to build")
endif()
find_package(Threads REQUIRED)

# Optional deps (header-only JSON)
set(NLOHMANN_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/include)
find_package(SQLite3 QUIET)

# Check if CUDA is available
if(TORCH_CUDA_AVAILABLE)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/audit/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(TRANSFORMER_SOURCES
    src/transformer_model.cpp
    src/feature_pipeline.cpp
    src/strategy_transformer.cpp
)

# Header files
set(TRANSFORMER_HEADERS
    include/sentio/transformer_strategy_core.hpp
    include/sentio/transformer_model.hpp
    include/sentio/feature_pipeline.hpp
    include/sentio/online_trainer.hpp
    include/sentio/strategy_transformer.hpp
)

# Create the main library
add_library(sentio_transformer STATIC ${TRANSFORMER_SOURCES} ${TRANSFORMER_HEADERS})

# Link libraries
target_link_libraries(sentio_transformer 
    ${TORCH_LIBRARIES}
    Threads::Threads
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sentio_transformer OpenMP::OpenMP_CXX)
endif()
if(HAVE_YAML)
    target_link_libraries(sentio_transformer yaml-cpp)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sentio_transformer PRIVATE
        -Wall -Wextra -Wpedantic
        -ffast-math
        -march=native
        -mtune=native
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(sentio_transformer PRIVATE
            -funroll-loops
            -fomit-frame-pointer
        )
    endif()
endif()

# MSVC specific flags
if(MSVC)
    target_compile_options(sentio_transformer PRIVATE
        /W4 /O2 /arch:AVX2
    )
endif()

# Define preprocessor macros
target_compile_definitions(sentio_transformer PRIVATE
    SENTIO_TRANSFORMER_ENABLED
    TORCH_EXTENSION_NAME=sentio_transformer
)

# Set target properties
set_target_properties(sentio_transformer PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${TRANSFORMER_HEADERS}"
)

# Installation rules
install(TARGETS sentio_transformer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/sentio
)

# Install configuration files
install(FILES 
    config/transformer_config.yaml
    config/production_config.yaml
    DESTINATION share/sentio/config
)

# ==================== Core Library ====================

# Aggregate commonly used Sentio modules for CLIs
add_library(sentio_core STATIC
    src/runner.cpp
    src/canonical_evaluation.cpp
    src/unified_metrics.cpp
    src/canonical_metrics.cpp
    src/feature_feeder.cpp
    src/strategy_profiler.cpp
    src/circuit_breaker.cpp
    src/execution_verifier.cpp
    src/adaptive_eod_manager.cpp
    src/adaptive_allocation_manager.cpp
    src/universal_position_coordinator.cpp
    src/future_qqq_loader.cpp
    src/mars_data_loader.cpp
    src/csv_loader.cpp
    src/data_downloader.cpp
    src/feature_cache.cpp
    src/time_utils.cpp
    src/signal_engine.cpp
    src/signal_pipeline.cpp
    src/router.cpp
    src/base_strategy.cpp
    src/strategy_signal_or.cpp
    src/strategy_initialization.cpp
    src/strategy_tfa.cpp
    src/ml/ts_model.cpp
    src/ml/model_registry_ts.cpp
    src/global_leverage_config.cpp
    src/leverage_aware_csv_loader.cpp
    src/leverage_pricing.cpp
    src/accurate_leverage_pricing.cpp
    src/feature_engineering/feature_normalizer.cpp
    src/feature_engineering/technical_indicators.cpp
    src/feature_engineering/kochi_features.cpp
    src/virtual_market_shim.cpp
    src/polygon_client.cpp
    src/audit.cpp
    audit/src/audit_db.cpp
    audit/src/audit_db_recorder.cpp
    audit/src/clock.cpp
    audit/src/hash.cpp
)
target_link_libraries(sentio_core sentio_transformer Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sentio_core OpenMP::OpenMP_CXX)
endif()
if(HAVE_YAML)
    target_link_libraries(sentio_core yaml-cpp)
endif()
# Link core deps
if(CURL_FOUND)
    target_link_libraries(sentio_core CURL::libcurl)
else()
    target_link_libraries(sentio_core curl)
endif()
if(SQLite3_FOUND)
    target_link_libraries(sentio_core SQLite::SQLite3)
endif()

# ==================== Executables ====================

# Training executable
# sentio_cli main
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
    add_executable(sentio_cli 
        src/main.cpp 
        src/cli_helpers.cpp 
        src/run_id_generator.cpp 
    )
    target_link_libraries(sentio_cli sentio_core)
    target_include_directories(sentio_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/audit/include)
    if(SQLite3_FOUND)
        target_link_libraries(sentio_cli SQLite::SQLite3)
    endif()
endif()

# Build the trainer main if present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformer_trainer_main.cpp)
    add_executable(transformer_trainer_main src/transformer_trainer_main.cpp)
    target_link_libraries(transformer_trainer_main sentio_transformer)
endif()

# Build evaluator
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/transformer_evaluator_main.cpp)
    add_executable(transformer_evaluator_main src/transformer_evaluator_main.cpp)
    target_link_libraries(transformer_evaluator_main sentio_transformer)
endif()

# ==================== Testing ====================

option(BUILD_TESTS "Build test suite" ON)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    if(NOT GTest_FOUND)
        message(WARNING "GTest not found; skipping tests")
        set(BUILD_TESTS OFF)
    endif()
    
    # Test sources
    set(TEST_SOURCES
        tests/transformer/test_transformer_model.cpp
        tests/transformer/test_feature_pipeline.cpp
        tests/transformer/test_online_trainer.cpp
        tests/transformer/test_strategy.cpp
        tests/transformer/test_performance.cpp
    )
    
    # Create test executable
    if(BUILD_TESTS)
        # Assemble tests; include YAML-dependent ones only if yaml-cpp is available
        set(TEST_SOURCES
            tests/transformer/test_transformer_model.cpp
            tests/transformer/test_feature_pipeline.cpp
        )
        if(HAVE_YAML)
            list(APPEND TEST_SOURCES tests/transformer/test_strategy.cpp)
        else()
            message(WARNING "Skipping test_strategy.cpp (requires yaml-cpp)")
        endif()

        add_executable(transformer_tests ${TEST_SOURCES})
        target_link_libraries(transformer_tests 
            sentio_transformer 
            GTest::GTest 
            GTest::Main
        )
        add_test(NAME TransformerTests COMMAND transformer_tests)
        # benchmark target disabled (requires Google Benchmark)
    endif()
endif()

# ==================== Documentation ====================

option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    
    set(DOXYGEN_INPUT_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in 
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# ==================== Package Configuration ====================

# Disable packaging config generation (template missing)
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SentioTransformerConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/SentioTransformerConfig.cmake"
#     INSTALL_DESTINATION lib/cmake/SentioTransformer
# )
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/SentioTransformerConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/SentioTransformerConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/SentioTransformerConfigVersion.cmake"
#     DESTINATION lib/cmake/SentioTransformer
# )

# ==================== Custom Targets ====================

# Format code
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i -style=file ${TRANSFORMER_SOURCES} ${TRANSFORMER_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting code with clang-format"
    )
endif()

# Static analysis
find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
    add_custom_target(tidy
        COMMAND ${CLANG_TIDY} ${TRANSFORMER_SOURCES} -- ${CMAKE_CXX_FLAGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running static analysis with clang-tidy"
    )
endif()

# Memory check
find_program(VALGRIND valgrind)
if(VALGRIND AND BUILD_TESTS)
    add_custom_target(memcheck
        COMMAND ${VALGRIND} --tool=memcheck --leak-check=full --show-leak-kinds=all 
                --track-origins=yes --verbose ./transformer_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS transformer_tests
        COMMENT "Running memory check with Valgrind"
    )
endif()

# Performance profiling
add_custom_target(profile
    COMMAND perf record -g ./benchmark_transformer
    COMMAND perf report
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS benchmark_transformer
    COMMENT "Profiling with perf"
)

# sentio_audit CLI
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/audit/src/audit_cli.cpp)
    add_executable(sentio_audit 
        audit/src/audit_cli.cpp 
        audit/src/audit_db.cpp 
        audit/src/audit_db_recorder.cpp 
        audit/src/clock.cpp 
        audit/src/hash.cpp
    )
    target_include_directories(sentio_audit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/audit/include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(sentio_audit sentio_core Threads::Threads)
    if(SQLite3_FOUND)
        target_link_libraries(sentio_audit SQLite::SQLite3)
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "======================= Configuration Summary =======================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "CUDA support: ${TORCH_CUDA_AVAILABLE}")
message(STATUS "OpenMP support: ${OpenMP_CXX_FOUND}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build docs: ${BUILD_DOCS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "====================================================================")
message(STATUS "")
