cmake_minimum_required(VERSION 2.8.8)

project (ex_lenet)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(ex_lenet_mnist ex_lenet_mnist.cpp mnist.cpp)
target_link_libraries(ex_lenet_mnist ${Caffe_LIBRARIES})
