cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

add_executable(simple_maze simple_maze.cpp)
add_executable(wind_and_islands wind_and_islads.cpp)

find_package(Python3 COMPONENTS Development NumPy)

target_include_directories(simple_maze PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
target_link_libraries(simple_maze Python3::Python Python3::NumPy)

target_link_libraries(wind_and_islands PUBLIC matplot)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


