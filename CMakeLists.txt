cmake_minimum_required(VERSION 2.8.12)

project(dlib_nap_detection)

find_package(OpenCV REQUIRED)

add_subdirectory(dlib)

set(source dlib_nap_project.cpp dark.c)

add_executable(dlib_nap_detection ${source})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dlib_nap_detection ${CMAKE_CURRENT_SOURCE_DIR}/libdarknet.so MVSDK ${OpenCV_LIBS} dlib::dlib pthread)


