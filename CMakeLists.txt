cmake_minimum_required(VERSION 3.9)
project(OpenGLDemo)


set(CMAKE_CXX_STANDARD 14)

find_package(glew REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

message(OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS})

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(3rdparty/sdm/include)



set(eos_3RDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty") # this is used by the CMakeLists.txt files in the subdirectories

set(CMAKE_MODULE_PATH ${eos_3RDPARTY_DIR}/eigen/cmake)
# Use the Eigen included in the submodule by default, but allow the user to override EIGEN3_INCLUDE_DIR:
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR ${eos_3RDPARTY_DIR}/eigen)
endif()
find_package(Eigen3 REQUIRED)

set(CEREAL_INCLUDE_DIR "${eos_3RDPARTY_DIR}/cereal/include")
set(glm_INCLUDE_DIR "${eos_3RDPARTY_DIR}/glm")
set(nanoflann_INCLUDE_DIR "${eos_3RDPARTY_DIR}/nanoflann/include")
set(eigen3_nnls_INCLUDE_DIR "${eos_3RDPARTY_DIR}/eigen3-nnls/src")
set(toml11_INCLUDE_DIR "${eos_3RDPARTY_DIR}/toml11")

include_directories("3rdparty/eos/include")
include_directories(${CEREAL_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${glm_INCLUDE_DIR})
include_directories(${nanoflann_INCLUDE_DIR})
include_directories(${eigen3_nnls_INCLUDE_DIR})
include_directories(${toml11_INCLUDE_DIR})
include_directories(eos_sample)

add_executable(EosImage eos_sample/eos_image_show.cpp  eos_sample/Eos.cpp)

target_link_libraries(EosImage  glfw  ${OPENGL_gl_LIBRARY}  ${GLEW_LIBRARIES} ${OpenCV_LIBS})