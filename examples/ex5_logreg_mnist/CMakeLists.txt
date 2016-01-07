cmake_minimum_required(VERSION 2.8.8)

project (ex_logreg)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(ex_logreg_mnist ex_logreg_mnist.cpp)
target_link_libraries(ex_logreg_mnist ${Caffe_LIBRARIES})