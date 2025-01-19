cmake_minimum_required(VERSION 3.16)
project(TT-GoL CXX)

option(USE_LIBCPP OFF)
set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

set(NORMALIZED_ARCH_NAME $ENV{ARCH_NAME})
if("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    set(NORMALIZED_ARCH_NAME "wormhole")
endif()

if(NOT DEFINED CPM_SOURCE_CACHE)
    message(STATUS "Setting CPM_SOURCE_CACHE to $ENV{TT_METAL_HOME}/.cpmcache")
    set(CPM_SOURCE_CACHE "$ENV{TT_METAL_HOME}/.cpmcache")
endif()
message(STATUS "CPM_SOURCE_CACHE is set to: ${CPM_SOURCE_CACHE}")

list(PREPEND CMAKE_MODULE_PATH "$ENV{TT_METAL_HOME}/cmake")
include(CPM)

if(CMAKE_VERSION VERSION_LESS 3.25)
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies EXCLUDE_FROM_ALL SYSTEM)
endif()

add_executable(TT-GoL main.cpp)

set_target_properties(TT-GoL PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_definitions(TT-GoL PRIVATE
    FMT_HEADER_ONLY
)

target_compile_options(TT-GoL PRIVATE
    -Werror
    -Wdelete-non-virtual-dtor
    -Wreturn-type
    -Wswitch
    -Wuninitialized
    -Wno-unused-parameter
    -mavx2
    -fPIC
    -fvisibility-inlines-hidden
    -fno-lto
    # Clang-specific options
    $<$<CXX_COMPILER_ID:Clang>:-Wsometimes-uninitialized>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-error=local-type-template-args>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-delete-non-abstract-non-virtual-dtor>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c99-designator>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-shift-op-parentheses>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-non-c-typedef-for-linkage>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-this-capture>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-builtins>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>
    # GCC-specific options
    $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overread>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-requires>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-error=non-template-friend>
)

find_package(SDL2 REQUIRED)

target_include_directories(TT-GoL PRIVATE
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/${NORMALIZED_ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/${NORMALIZED_ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/detail
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon
    $ENV{TT_METAL_HOME}/tt_metal/include
    $ENV{TT_METAL_HOME}/tt_metal/third_party/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/taskflow/3rd-party/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
)

target_link_directories(TT-GoL PRIVATE
    $ENV{TT_METAL_HOME}/build/lib
)

target_link_libraries(TT-GoL PRIVATE
    fmt  
    magic_enum
    Reflect::Reflect
    yaml-cpp
    Boost::core
    Boost::smart_ptr
    Boost::container
    tt_metal
    SDL2::SDL2
    $ENV{TT_METAL_HOME}/build/lib/libdevice.so
)

target_compile_options(TT-GoL PRIVATE -stdlib=libc++)
target_link_options(TT-GoL PRIVATE -lc++ -lc++abi)