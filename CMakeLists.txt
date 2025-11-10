cmake_minimum_required(VERSION 3.16)
project(drake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

# Try to find pybind11 first; if not present, fetch it so projects can configure
# without requiring a system package.
find_package(pybind11 CONFIG QUIET)

# Uncomment if you want to build example codes.
# add_subdirectory(example)
add_subdirectory(src)
