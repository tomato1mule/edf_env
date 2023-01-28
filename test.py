import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_name = "arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

move_group.set_pose_reference_frame('map')

# # We can get the name of the reference frame for this robot:
# planning_frame = move_group.get_planning_frame()
# print("============ Planning frame: %s" % planning_frame)

# # We can also print the name of the end-effector link for this group:
# eef_link = move_group.get_end_effector_link()
# print("============ End effector link: %s" % eef_link)

# # We can get a list of all the groups in the robot:
# group_names = robot.get_group_names()
# print("============ Available Planning Groups:", robot.get_group_names())

# # Sometimes for debugging it is useful to print the entire state of the
# # robot:
# print("============ Printing robot state")
# print(robot.get_current_state())
# print("")



pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = 0.0
pose_goal.orientation.y = 1.0
pose_goal.orientation.z = 0.0
pose_goal.orientation.w = 0.0
pose_goal.position.x = 0.0
pose_goal.position.y = 0.0
pose_goal.position.z = 0.75

move_group.set_pose_target(pose_goal)

success = move_group.go(wait=True)
move_group.stop()
move_group.clear_pose_targets()