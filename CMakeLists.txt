cmake_minimum_required(VERSION 3.20)
project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

option(TINYINFER_NATIVE "Tune for the build machine's CPU" ON)
if(TINYINFER_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
  if(HAS_MCPU_NATIVE)
    add_compile_options(-mcpu=native)
  else()
    add_compile_options(-march=native)
  endif()
endif()

option(TINYINFER_OPENMP "Multithread kernels with OpenMP" ON)
if(TINYINFER_OPENMP)
  if(APPLE AND NOT OpenMP_CXX_FLAGS)
    # Apple clang ships without OpenMP; use Homebrew's libomp.
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(LIBOMP_PREFIX)
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
  endif()
  find_package(OpenMP)
endif()

add_library(tinyinfer_core STATIC
  src/version.cpp
  src/tensor.cpp
  src/tinf.cpp
  src/gpt2.cpp
  src/ops.cpp
  src/matmul.cpp
  src/kernels.cpp
  src/attention.cpp
  src/tokenizer.cpp
  src/generate.cpp
  src/sampling.cpp
  src/top_p.cpp
  src/kv_cache.cpp
  src/model.cpp
  src/llama.cpp
  src/rope.cpp
)
target_include_directories(tinyinfer_core PUBLIC src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tinyinfer_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(tinyinfer src/main.cpp)
target_link_libraries(tinyinfer PRIVATE tinyinfer_core)

set(TINYINFER_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/gpt2" CACHE PATH "Exported GPT-2 directory")
set(TINYINFER_LLAMA_DIR "${CMAKE_SOURCE_DIR}/models/qwen2.5-0.5b" CACHE PATH "Exported Llama-style model directory")

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/test_*.cpp)
foreach(src ${TEST_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE tinyinfer_core)
  target_compile_definitions(${name} PRIVATE TINYINFER_MODEL_DIR="${TINYINFER_MODEL_DIR}"
                                             TINYINFER_LLAMA_DIR="${TINYINFER_LLAMA_DIR}")
  add_test(NAME ${name} COMMAND ${name})
endforeach()

file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench/bench_*.cpp)
foreach(src ${BENCH_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE tinyinfer_core)
endforeach()
