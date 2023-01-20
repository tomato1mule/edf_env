import threading
from typing import Optional

import numpy as np

import rospy
import actionlib
from ros_numpy.point_cloud2 import array_to_pointcloud2

from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory

from edf_env.env import UR5Env
from edf_env.pc_utils import encode_pc







class UR5EnvRosWrapper():
    def __init__(self, env: UR5Env):
        self.env: UR5Env = env

        self.current_traj: Optional[JointTrajectory] = None

        self.joint_pub = rospy.Publisher('joint_states', JointState, latch=False, queue_size=10)
        self.scene_pc_pub = rospy.Publisher('scene_pointcloud', PointCloud2, latch=True)
        self.arm_ctrl_AS_name = 'arm_controller/follow_joint_trajectory'
        self.arm_ctrl_AS = actionlib.SimpleActionServer(self.arm_ctrl_AS_name, FollowJointTrajectoryAction,
                                                              execute_cb = self.execute_cb, auto_start = False)                                                           
        rospy.init_node('edf_env', anonymous=True, log_level=rospy.INFO)
        self.arm_ctrl_AS.start()


        self.threads=[]
        jointpub_thread = threading.Thread(name='jointpub_thread', target=self.jointpub_thread)
        self.threads.append(jointpub_thread)


        self.publish_scene_pc()
        for thread in self.threads:
            thread.start()

    def publish_scene_pc(self):
        stamp = rospy.Time.now()
        frame_id = 'map'

        points, colors = self.env.observe_scene_pc()
        pc = encode_pc(points=points, colors=colors)
        msg: PointCloud2 = array_to_pointcloud2(cloud_arr = pc, stamp=stamp, frame_id=frame_id)
        self.scene_pc_pub.publish(msg)

    def close(self):
        self.env.close()
        for thread in self.threads:
            thread.terminate()


    def jointpub_thread(self):
        # TODO:  https://stackoverflow.com/questions/50907224/not-able-to-terminate-the-process-in-multiprocessing-python-linux
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            self.publish_joint_info()
            rate.sleep()
        

    def publish_joint_info(self):
        pos, vel = self.env.get_joint_states()

        header = Header()
        header.stamp = rospy.Time.now()

        msg = JointState()
        msg.header = header
        for id in range(self.env.n_joints):
            msg.name.append(self.env.robot_joint_dict[id])
            msg.position.append(pos[id])
            msg.velocity.append(vel[id])
        self.joint_pub.publish(msg)

    def execute_cb(self, goal):
        trajectory: JointTrajectory = goal.trajectory
        path_tolerance: JointTolerance = goal.path_tolerance
        goal_tolerance: JointTolerance = goal.goal_tolerance
        duration: Duration = goal.goal_time_tolerance
        joint_names: list[str] = trajectory.joint_names


        #r = rospy.Rate(10)
        success = True
        target_time_from_start: float = 0.
        for point in trajectory.points:
            # check that preempt has not been requested by the client
            if self.arm_ctrl_AS.is_preempt_requested():
                rospy.logwarn(f"{self.arm_ctrl_AS_name}: Preempted")
                self.arm_ctrl_AS.set_preempted()
                success = False
                break

            target_pos: list[float] = point.positions
            target_vel: list[float] = point.velocities
            target_acc: list[float] = point.accelerations
            target_effort: list[float] = point.effort
            target_duration = point.time_from_start.to_sec() - target_time_from_start
            target_time_from_start: float = point.time_from_start.to_sec()

            self.env.control_target_joint_states(target_pos=target_pos, target_vel=target_vel, target_duration=target_duration, target_joint_names=joint_names)

            # publish the feedback
            header = Header()
            header.stamp = rospy.Time.now()
            feedback = FollowJointTrajectoryFeedback()
            feedback.header = header
            self.arm_ctrl_AS.publish_feedback(feedback)
            #r.sleep()



        if success:
            result = FollowJointTrajectoryResult()
            rospy.loginfo(f"{self.arm_ctrl_AS_name}: Succeeded")
            self.arm_ctrl_AS.set_succeeded(result)




