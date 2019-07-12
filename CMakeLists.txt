cmake_minimum_required(VERSION 2.8)
project( webcam-ergonomics )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )



#set(JSON_BuildTests OFF CACHE INTERNAL "")
#add_subdirectory(json-develop)
#add_library(foo ...)
#target_link_libraries(foo PRIVATE nlohmann_json::nlohmann_json)



add_executable( webcam-ergonomics src/main.cpp )
target_link_libraries( webcam-ergonomics ${OpenCV_LIBS} )
