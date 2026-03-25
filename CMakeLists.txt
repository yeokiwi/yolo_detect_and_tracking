cmake_minimum_required(VERSION 3.15)
project(YoloDetectAndTracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV 4.x with required modules
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/yolov9_detector.cpp
    src/ncc_tracker.cpp
    src/multi_tracker.cpp
    src/utils.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/config
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Windows: copy OpenCV DLLs next to the executable
if(WIN32)
    foreach(_lib ${OpenCV_LIBS})
        if(TARGET ${_lib})
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${_lib}>
                    $<TARGET_FILE_DIR:${PROJECT_NAME}>
            )
        endif()
    endforeach()
endif()
