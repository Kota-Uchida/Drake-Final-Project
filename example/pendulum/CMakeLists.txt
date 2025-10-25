cmake_minimum_required(VERSION 3.16)
project(pendulum_meshcat)

find_package(Drake REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE drake::drake)

