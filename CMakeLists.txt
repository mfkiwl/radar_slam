cmake_minimum_required(VERSION 3.1)
project(yeti)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")
#set(CMAKE_CXX_STANDARD 14)
find_package(catkin REQUIRED COMPONENTS
		roscpp
		std_msgs
		nav_msgs
		sensor_msgs
		geometry_msgs
		)

catkin_package()

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
find_package(PkgConfig)

find_package(PythonLibs REQUIRED)

include_directories(
		include
    	third_party/matplotlib-cpp
		third_party/nanoflann
		third_party/teaser
		${catkin_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
		${catkin_INCLUDE_DIRS}
#    ${EIGEN_INCLUDE_DIR}
		/usr/local/include/eigen3/
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_library(features
  src/features.cpp
)

add_library(teaser
		third_party/teaser/src/certification.cc
		third_party/teaser/src/graph.cc
		third_party/teaser/src/matcher.cc
		third_party/teaser/src/registration.cc
		)

target_link_libraries(features
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
#	${Boost_INCLUDE_DIRS}
		${Boost_LIBRARIES}
#    Eigen3::Eigen
)

add_library(radar_utils
  src/radar_utils.cpp
)

target_link_libraries(radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
#	${Boost_INCLUDE_DIRS}
		${Boost_LIBRARIES}
#    Eigen3::Eigen
)

add_library(association
  src/association.cpp
)

target_link_libraries(association
	${catkin_LIBRARIES}
#	${Boost_INCLUDE_DIRS}
		${Boost_LIBRARIES}
#    Eigen3::Eigen
)

#add_executable(visualization src/visualization.cpp)
#target_link_libraries(visualization
#    features
#    radar_utils
#	${catkin_LIBRARIES}
#	${OpenCV_LIBRARIES}
#		${Boost_LIBRARIES}
##	${Boost_INCLUDE_DIRS}
#)

#target_include_directories(visualization /usr/include)
add_executable(odometry src/odometry.cpp)
target_link_libraries(odometry
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)

#add_executable(odometry_node src/odometry_ros.cpp)
#target_link_libraries(odometry_node
#		features
#		radar_utils
#		association
#		${catkin_LIBRARIES}
#		${OpenCV_LIBRARIES}
#		)

#add_executable(pcl_match src/pcl_match.cpp)
#target_link_libraries(pcl_match
#		features
#		radar_utils
#		association
#		${catkin_LIBRARIES}
#		${OpenCV_LIBRARIES}
#		teaser
##		teaser_lib
#		)

#add_executable(mytest src/test.cpp)
#target_link_libraries(mytest
#		features
#		radar_utils
#		association
#		${catkin_LIBRARIES}
#		${OpenCV_LIBRARIES}
#		)

#add_executable(raw_match src/raw_match.cpp)
#target_link_libraries(raw_match
#		features
#		radar_utils
#		association
#		${OpenCV_LIBRARIES}
#		)

add_library(matplotlibcpp INTERFACE)
target_include_directories(matplotlibcpp INTERFACE include)
target_include_directories(matplotlibcpp INTERFACE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(matplotlibcpp INTERFACE ${PYTHON_LIBRARIES})

#add_executable(test_motiondistortion src/test_motiondistortion.cpp)
#target_link_libraries(test_motiondistortion
#    association
#	${catkin_LIBRARIES}
#	${OpenCV_LIBRARIES}
#	matplotlibcpp
#)
#target_compile_definitions(test_motiondistortion PRIVATE "-DWITHOUT_NUMPY")

#add_executable(showcase_distortion src/showcase_distortion.cpp)
#target_link_libraries(showcase_distortion
#    features
#    radar_utils
#    association
#	${catkin_LIBRARIES}
#	${OpenCV_LIBRARIES}
#	matplotlibcpp
#)
#target_compile_definitions(showcase_distortion PRIVATE "-DWITHOUT_NUMPY")
#
add_executable(localization src/localization.cpp)
target_link_libraries(localization
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)
