cmake_minimum_required(VERSION 3.16)
project(drake_final_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG QUIET)

add_subdirectory(cpp)

