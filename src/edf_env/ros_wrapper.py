import threading
from typing import Optional

import rospy
import actionlib

from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory

from edf_env.env import UR5Env

class UR5EnvRosWrapper():
    def __init__(self, env: UR5Env):
        self.env: UR5Env = env

        self.current_traj: Optional[JointTrajectory] = None

        self.joint_pub = rospy.Publisher('joint_states', JointState, latch=False, queue_size=10)
        self.arm_ctrl_AS_name = 'arm_controller/follow_joint_trajectory'
        self.arm_ctrl_AS = actionlib.SimpleActionServer(self.arm_ctrl_AS_name, FollowJointTrajectoryAction,
                                                              execute_cb = self.execute_cb, auto_start = False)
        rospy.init_node('edf_env', anonymous=True)
        self.arm_ctrl_AS.start()


        self.threads=[]
        b = threading.Thread(name='background', target=self.background)
        self.threads.append(b)

        for thread in self.threads:
            thread.start()


    def close(self):
        self.env.close()
        for thread in self.threads:
            thread.terminate()


    def background(self):
        # TODO:  https://stackoverflow.com/questions/50907224/not-able-to-terminate-the-process-in-multiprocessing-python-linux
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            self.publish_joint_info()
            rate.sleep()
        

    def publish_joint_info(self):
        pos, vel = self.env.get_joint_states(list(range(self.env.n_joints)))

        header = Header()
        header.stamp = rospy.Time.now()

        # for id in self.movable_joints_id:
        #     msg = JointState()
        #     msg.header = header
        #     msg.name = self.env.robot_joint_name_list[id]
        #     msg.position = [pos[id]]
        #     msg.velocity = [vel[id]]
        #     self.joint_pub.publish(msg)
        msg = JointState()
        msg.header = header
        for id in self.env.movable_joints_id:
            msg.name.append(self.env.robot_joint_name_list[id])
            msg.position.append(pos[id])
            msg.velocity.append(vel[id])
        self.joint_pub.publish(msg)

    def execute_cb(self, goal):
        rospy.logerr(f"CB Acted")
        trajectory: JointTrajectory = goal.trajectory
        path_tolerance: JointTolerance = goal.path_tolerance
        goal_tolerance: JointTolerance = goal.goal_tolerance
        duration: Duration = goal.goal_time_tolerance
        r = rospy.Rate(1)

        success = True
        for i in range(2):
            # check that preempt has not been requested by the client
            if self.arm_ctrl_AS.is_preempt_requested():
                rospy.logwarn(f"{self.arm_ctrl_AS_name}: Preempted")
                self.arm_ctrl_AS.set_preempted()
                success = False
                break

            # publish the feedback
            header = Header()
            header.stamp = rospy.Time.now()
            feedback = FollowJointTrajectoryFeedback()
            feedback.header = header
            self.arm_ctrl_AS.publish_feedback(feedback)
            r.sleep()

        if success:
            result = FollowJointTrajectoryResult()
            rospy.loginfo(f"{self.arm_ctrl_AS_name}: Succeeded")
            self.arm_ctrl_AS.set_succeeded(result)




