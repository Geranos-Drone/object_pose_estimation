cmake_minimum_required(VERSION 3.0.2)
project(object_pose_estimation)


find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
	DIRECTORY msg
	FILES
	SingleKP.msg
	KeyPoints.msg
)

generate_messages(
	DEPENDENCIES std_msgs geometry_msgs
)

catkin_package(
	CATKIN_DEPENDS std_msgs geometry_msgs message_runtime
)



