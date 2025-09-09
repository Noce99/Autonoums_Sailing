cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

add_executable(model model.cpp)
include_directories(../Sockets)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python3 COMPONENTS Development NumPy)

target_link_libraries(model Eigen3::Eigen)

target_include_directories(model PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
target_link_libraries(model Python3::Python Python3::NumPy)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


