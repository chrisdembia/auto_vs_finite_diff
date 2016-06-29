cmake_minimum_required(VERSION 3.0)
project(auto_vs_finite)

set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ADOLC REQUIRED adolc)
include_directories(${ADOLC_INCLUDE_DIRS})
link_directories(${ADOLC_LIBRARY_DIRS})

add_executable(auto_vs_finite auto_vs_finite.cpp)

target_link_libraries(auto_vs_finite ${ADOLC_LIBRARIES})
