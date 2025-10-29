cmake_minimum_required(VERSION 3.16)
project(drake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

# Try to find pybind11 first; if not present, fetch it so projects can configure
# without requiring a system package.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
	include(FetchContent)
	FetchContent_Declare(
		pybind11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		# Pin to a reasonably recent stable tag; change if you need a specific version
		GIT_TAG v2.11.0
	)
	FetchContent_MakeAvailable(pybind11)
endif()

add_subdirectory(pybind-example)
add_subdirectory(example)
add_subdirectory(src)
