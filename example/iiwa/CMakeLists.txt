cmake_minimum_required(VERSION 3.16)
project(iiwa_test)

find_package(Drake REQUIRED)

add_executable(iiwa_test iiwa.cpp)
target_link_libraries(iiwa_test PRIVATE drake::drake)