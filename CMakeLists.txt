cmake_minimum_required(VERSION 2.6)
project(visual_odometry)
SET(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


#find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories (
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
#${PCL_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library (${PROJECT_NAME} SHARED 
src/draw.cpp
src/frame.cpp
)

target_link_libraries(${PROJECT_NAME}
#${PCL_LIBRARIES} 
${OpenCV_LIBS}
)

# build main
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(vo src/main.cpp)
target_link_libraries(
vo 
${PROJECT_NAME} 
#${PCL_LIBRARIES} 
${OpenCV_LIBS}
)
