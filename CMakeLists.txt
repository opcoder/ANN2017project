cmake_minimum_required(VERSION 3.7)
project(CLionANN)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp net.cpp net.hpp layer.hpp layer.cpp utils.hpp utils.cpp)
add_executable(CLionANN ${SOURCE_FILES})