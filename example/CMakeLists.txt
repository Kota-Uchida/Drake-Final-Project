cmake_minimum_required(VERSION 3.16)

# Add "add.cpp" as python module called "add"
pybind11_add_module(add_cpp add_module.cpp)

# Refer to python module from C++ code
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_executable(sqrt sqrt_use.cpp)
# Link pybind11's embed target so <pybind11/embed.h> is available.
target_link_libraries(sqrt PRIVATE pybind11::embed Python3::Python)

# Determine Python site-packages directory at configure time so we can
# install the built pybind11 extension directly into the active Python
# environment and make it immediately importable.
execute_process(
	COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
	OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYTHON_SITE_PACKAGES)
	message(FATAL_ERROR "Failed to determine Python site-packages directory")
endif()

# Install the pybind11 module into the discovered site-packages so Python
# can import it right after `cmake --build . --target install`.
install(TARGETS add_cpp
	LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)

# Optionally install the executable into a bin directory under CMAKE_INSTALL_PREFIX
install(TARGETS sqrt RUNTIME DESTINATION bin OPTIONAL)
