cmake_minimum_required(VERSION 3.1)
project(Simple-HDR)

# In linux
# cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
set(CMAKE_CONFIGURATION_TYPES "Release")

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/source")
include_directories(${INCLUDE_DIR})

file(GLOB_RECURSE HEADER_DIR "./source/*.h")
file(GLOB_RECURSE SRC_DIR "./source/*.cpp")
add_executable(${PROJECT_NAME} ${HEADER_DIR} ${SRC_DIR})