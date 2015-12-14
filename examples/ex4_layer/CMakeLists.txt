project(argmax_test)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(argmax_test main.cpp)
target_link_libraries(argmax_test ${Caffe_LIBRARIES})




