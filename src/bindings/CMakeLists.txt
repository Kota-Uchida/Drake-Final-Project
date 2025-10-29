cmake_minimum_required(VERSION 3.16)

project(drake_bindings LANGUAGES CXX)

pybind11_add_module(drake_bindings bindings.cpp)

# If you have error finding <pybind11/embed.h>, uncomment the following lines
# target_link_libraries(drake_bindings PRIVATE pybind11::embed Python3::Python)

# Ensure the pybind11 module can find the local drake_core headers. Use
# ${CMAKE_SOURCE_DIR} (the repository root) so the path is correct regardless
# of the current subdirectory.
target_include_directories(drake_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/drake_core
)

# Place the built Python extension into python/drake so it can be imported
# as a local package during development.
set_target_properties(drake_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/drake"
)