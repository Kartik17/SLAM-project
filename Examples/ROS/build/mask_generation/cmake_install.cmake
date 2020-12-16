# Install script for directory: /media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/src/mask_generation

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/build/mask_generation/catkin_generated/installspace/mask_generation.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mask_generation/cmake" TYPE FILE FILES
    "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/build/mask_generation/catkin_generated/installspace/mask_generationConfig.cmake"
    "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/build/mask_generation/catkin_generated/installspace/mask_generationConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mask_generation" TYPE FILE FILES "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/src/mask_generation/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mask_generation" TYPE PROGRAM FILES "/media/sbpl/A69AFABA9AFA85D9/guru/fall2020/courses/slam/project/SLAM-project/ORB_SLAM2/Examples/ROS/build/mask_generation/catkin_generated/installspace/dynamic_detector_main.py")
endif()

