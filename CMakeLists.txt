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

add_library(tinyinfer_core STATIC
  src/version.cpp
  src/tensor.cpp
  src/tinf.cpp
  src/gpt2.cpp
  src/ops.cpp
  src/matmul.cpp
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
