cmake_minimum_required(VERSION 3.16)
project(drake_core LANGUAGES CXX)

add_library(drake_core core.cpp utils.cpp)
target_include_directories(drake_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
