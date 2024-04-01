# Simple CMakeLists.txt:

cmake_minimum_required(VERSION 3.10)

project(capricorngrad)

file(GLOB_RECURSE SOURCES "src/**/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "tests/**/*.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# optimization
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Qt5
find_package(Qt5 COMPONENTS Core Gui Charts Widgets REQUIRED)

add_executable(capricorngrad apps/main.cpp ${SOURCES})

add_executable(capricorngrad_test apps/capricorngrad_test.cpp ${SOURCES} ${TEST_SOURCES})

target_link_libraries(capricorngrad Qt5::Widgets Qt5::Charts)
target_link_libraries(capricorngrad_test Qt5::Widgets Qt5::Charts)

# boost
# find_package(Boost REQUIRED COMPONENTS program_options)