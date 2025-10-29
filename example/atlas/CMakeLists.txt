cmake_minimum_required(VERSION 3.16)
project(atlas_test)

find_package(Drake REQUIRED)

add_executable(atlas_test atlas.cpp)
target_link_libraries(atlas_test PRIVATE drake::drake)

