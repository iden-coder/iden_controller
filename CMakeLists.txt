cmake_minimum_required(VERSION 2.8.3)
project(iden_controller)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  message_generation
  genmsg
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  serial
  roslib
  nav_core
  pluginlib
  costmap_2d
)

## Generate messages in the 'msg' folder
# add_message_files(FILES
#   # Battery_info.msg
#   # Health_info.msg
#   # Robot_health.msg
# )

## Generate services in the 'srv' folder
add_service_files(FILES
  GetMaxVel.srv
  SetMaxVel.srv
  GetSensorTF.srv
  SetSensorTF.srv
  GetBatteryInfo.srv
  SetLEDMode.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
#  INCLUDE_DIRS include
#  LIBRARIES iden_controller
#  CATKIN_DEPENDS genmsg geometry_msgs roscpp rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
# include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(include)

# 统一使用 my_base_driver，通过 enable_pid 参数控制是否启用 PID
# base_driver_iden.cpp 保留为参考，不再独立编译
# add_executable(base_driver_iden  src/base_driver_iden.cpp)
# add_dependencies(base_driver_iden iden_controller_generate_messages_cpp)
# target_link_libraries(base_driver_iden iden_crc_table_fd ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_executable(iden_my_base_driver  src/my_base_driver_iden.cpp)
add_dependencies(iden_my_base_driver iden_controller_generate_messages_cpp)
target_link_libraries(iden_my_base_driver iden_crc_table_fd ${catkin_LIBRARIES} ${Boost_LIBRARIES})

## iden_crc_table_fd
add_library(iden_crc_table_fd src/crc_table_iden.cpp)

## ===== IdenPlanner: 高性能局部规划器插件 =====
add_library(iden_planner
  src/iden_planner.cpp
)
add_dependencies(iden_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(iden_planner
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## 安装插件描述文件
install(FILES
  iden_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## 安装 iden_planner 库
install(TARGETS iden_planner
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## ===== 单元测试 =====
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  find_package(cv_bridge REQUIRED)

  # IdenPlanner 核心算法测试
  catkin_add_gtest(test_iden_planner
    test/test_iden_planner.cpp
  )
  target_link_libraries(test_iden_planner
    iden_planner
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${cv_bridge_LIBRARIES}
  )
  add_dependencies(test_iden_planner ${${PROJECT_NAME}_EXPORTED_TARGETS})
endif()
