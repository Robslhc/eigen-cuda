######### CMake Version #####################
cmake_minimum_required(VERSION 3.12)
#############################################

######### Project Name ######################
project(eigencuda)
SET( EXECUTABLE_NAME eigen_cuda_test )
#############################################

######### Options ###########################
option( CORE_USE_CUDA    "Use CUDA to speed up certain parts of the code."              ON )
#############################################


######### CUDA decisions ####################
if (CORE_USE_CUDA)
    MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
    set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" )
    if (APPLE OR UNIX)
        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    elseif (WIN32)
        ### By default we use VS
        MESSAGE( STATUS ">> User compiler:             MSVC" )
        MESSAGE( STATUS ">> Choosing a different compiler is not yet implemented for Windows" )
    endif()
endif()
#############################################


######### Info ##############################
MESSAGE( STATUS ">> CMAKE_C_COMPILER:        " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:      " ${CMAKE_CXX_COMPILER} )
#############################################



### Find includes in corresponding build directories
set( CMAKE_INCLUDE_CURRENT_DIR ON )
### Output paths for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_SOURCE_DIR} )
    # set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${youroutputdirectory} )
    # set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${youroutputdirectory} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )


######### CUDA decisions ####################
if (CORE_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
endif()
#############################################

#############################################
set(SOURCE_FILES
    src/test.cpp
    src/kernel.cu
    src/kernel.cpp
)

set(HEADER_FILES
    include/test.hpp
    include/kernel.hpp
)
#############################################



#############################################
if (CORE_USE_CUDA)
    cuda_add_executable( ${EXECUTABLE_NAME} main.cpp ${SOURCE_FILES} )
else()
    add_executable( ${EXECUTABLE_NAME} main.cpp ${SOURCE_FILES} )
endif()

target_link_libraries( ${EXECUTABLE_NAME} ${CUDA_LIBRARIES})

target_include_directories( ${EXECUTABLE_NAME} PUBLIC include)
target_include_directories( ${EXECUTABLE_NAME} PRIVATE thirdparty)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_EXTENSIONS OFF)
#############################################