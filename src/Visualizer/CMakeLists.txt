cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

add_executable(visualizer visualizer.cpp)
add_executable(world_visualizer world_visualizer.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python3 COMPONENTS Development NumPy)
find_package(OpenCV REQUIRED)
find_library(soil2 REQUIRED)

target_link_libraries(visualizer Eigen3::Eigen)

target_include_directories(visualizer PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
target_link_libraries(visualizer Python3::Python Python3::NumPy)
find_package(glfw3 3.3 REQUIRED)

include_directories(/usr/include/GL/)
include_directories(../Utils)
include_directories(../Sockets)
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(GLEW_LIB NAMES glew)
find_library(GL_LIB NAMES GL)
target_link_libraries(visualizer glfw)
target_link_libraries(visualizer ${GLEW_LIB})
target_link_libraries(visualizer ${GL_LIB})

target_link_libraries(world_visualizer glfw)
target_link_libraries(world_visualizer ${GLEW_LIB})
target_link_libraries(world_visualizer ${GL_LIB})
target_link_libraries(world_visualizer ${OpenCV_LIBS})
target_link_libraries(world_visualizer ${soil2})

MACRO(ADDOBJ curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*.obj)
  SET(objlist "")
  FOREACH(child ${children})
  	LIST(APPEND objlist ${child})
  ENDFOREACH()
  MESSAGE(STATUS "OBJs:")
  FOREACH(obj ${objlist})
  	MESSAGE(STATUS "\t" ${obj})
  	configure_file(${curdir}/${obj} ${PROJECT_BINARY_DIR}/bin/ COPYONLY)
  ENDFOREACH()
ENDMACRO()

MACRO(ADDGLSL curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*.glsl)
  SET(glsllist "")
  FOREACH(child ${children})
  	LIST(APPEND glsllist ${child})
  ENDFOREACH()
  MESSAGE(STATUS "GLSLs:")
  FOREACH(glsl ${glsllist})
  	MESSAGE(STATUS "\t" ${glsl})
  	configure_file(${curdir}/${glsl} ${PROJECT_BINARY_DIR}/bin/shaders COPYONLY)
  ENDFOREACH()
ENDMACRO()

#ADDOBJ(${CMAKE_CURRENT_SOURCE_DIR}/../../resources/objs)
ADDGLSL(${CMAKE_CURRENT_SOURCE_DIR}/shaders)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


