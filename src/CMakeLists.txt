FIND_PACKAGE( PCL REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED )

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )
# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

#==build library for my own source code=====
add_library (${PROJECT_NAME} SHARED 
draw.cpp
frame.cpp
map.cpp
optimizer.cpp
ORBextractor.cc
)

target_link_libraries(${PROJECT_NAME}
${PCL_LIBRARIES} 
${OpenCV_LIBS}
${EIGEN3_LIBS}
g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY}
)

#=======build main=========================
ADD_EXECUTABLE(vo main.cpp )
TARGET_LINK_LIBRARIES( 
vo 
${PROJECT_NAME} 
${OpenCV_LIBS}
${PCL_LIBRARIES}
)
