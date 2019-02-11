cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)     
find_package(OpenCV REQUIRED)   

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(example-app main.cpp)
# link_directories(/usr/local/lib)   'find_package' has already done this
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)