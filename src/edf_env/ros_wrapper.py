import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any

import numpy as np

import rospy
import actionlib
import tf2_ros
from ros_numpy.point_cloud2 import array_to_pointcloud2
from ros_numpy.image import numpy_to_image

import moveit_commander

from sensor_msgs.msg import JointState, PointCloud2, Image
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, Pose


from edf_env.env import UR5Env
from edf_env.pc_utils import encode_pc
from edf_env.interface import EdfInterface
from edf_env.utils import CamData







class UR5EnvRosHandle():
    def __init__(self, env: UR5Env, monitor_refresh_rate: float = 0):
        self.env: UR5Env = env

        self.current_traj: Optional[JointTrajectory] = None
        self.scene_points: Optional[np.ndarray] = None
        self.scene_colors: Optional[np.ndarray] = None
        self.scene_pc_msg: Optional[PointCloud2] = None 

        
        self.joint_pub = rospy.Publisher('joint_states', JointState, latch=False, queue_size=10)
        self.scene_pc_pub = rospy.Publisher('scene_pointcloud', PointCloud2, latch=False, queue_size=1)
        self.base_link_tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.arm_ctrl_AS_name = 'arm_controller/follow_joint_trajectory'
        self.arm_ctrl_AS = actionlib.SimpleActionServer(self.arm_ctrl_AS_name, FollowJointTrajectoryAction,
                                                              execute_cb = self.execute_cb, auto_start = False)                                                           
        
        self.monitor_refresh_rate = monitor_refresh_rate
        self.monitor_img_pubs = []
        for i in range(len(self.env.monitor_cam_configs)):
            self.monitor_img_pubs.append(rospy.Publisher(f"monitor_img_{i}", Image, latch=False, queue_size=10))

        moveit_commander.roscpp_initialize(sys.argv)

        rospy.init_node('edf_env', anonymous=True, log_level=rospy.INFO)
        self.arm_ctrl_AS.start()


        self.threads=[]
        self.threads.append(threading.Thread(name='jointpub_thread', target=self.jointpub_thread))
        self.threads.append(threading.Thread(name='tfpub_thread', target=self.tfpub_thread))
        self.threads.append(threading.Thread(name='pcpub_thread', target=self.pcpub_thread))
        if self.monitor_refresh_rate > 0:
            self.threads.append(threading.Thread(name='monitor_imgpub_thread', target=self.monitor_imgpub_thread))


        self.update_scene_pc_msg()
        for thread in self.threads:
            thread.start()

        time.sleep(1)
        self.robot_com = moveit_commander.RobotCommander()
        self.scene_intf = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.arm_group.set_planner_id("BiTRRT")
        self.arm_group.set_planning_time(0.5)
        self.arm_group.set_pose_reference_frame('map')

        self.gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        self.gripper_group.set_planning_time(0.5)


    def close(self):
        self.env.close()
        for thread in self.threads:
            thread.terminate()

    def monitor_imgpub_thread(self):
        rate = rospy.Rate(self.monitor_refresh_rate)
        while not rospy.is_shutdown():
            imgs = [cam_dat['color'] for cam_dat in self.env.observe_monitor_img(return_seg=False, color_encoding='uint8')]
            for img, pub in zip(imgs, self.monitor_img_pubs):
                self.publish_image(img=img, pub=pub)
            rate.sleep()

    def jointpub_thread(self):
        # TODO:  https://stackoverflow.com/questions/50907224/not-able-to-terminate-the-process-in-multiprocessing-python-linux
        rate = rospy.Rate(20) # 10hz
        while not rospy.is_shutdown():
            self.publish_joint_info()
            rate.sleep()

    def tfpub_thread(self):
        rate = rospy.Rate(20) 
        while not rospy.is_shutdown():
            self.publish_base_link_tf()
            rate.sleep()

    def pcpub_thread(self):
        rate = rospy.Rate(2) 
        while not rospy.is_shutdown():
            self.publish_scene_pc()
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

    def update_scene_pc_msg(self):
        stamp = rospy.Time.now()
        frame_id = 'map'

        self.scene_points, self.scene_colors = self.env.observe_scene_pc()
        pc = encode_pc(points=self.scene_points, colors=self.scene_colors)
        self.scene_pc_msg: PointCloud2 = array_to_pointcloud2(cloud_arr = pc, stamp=stamp, frame_id=frame_id)

    def publish_scene_pc(self):
        if self.scene_pc_msg is None:
            self.update_scene_pc_msg()
        stamp = rospy.Time.now()
        self.scene_pc_msg.header.stamp = stamp
        self.scene_pc_pub.publish(self.scene_pc_msg)

    def publish_base_link_tf(self):
        pos, orn = self.env.get_base_pose()
        
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x,  t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = orn

        self.base_link_tf_broadcaster.sendTransform(t)

    def publish_image(self, img: np.ndarray, pub: rospy.Publisher):
        msg = numpy_to_image(arr=img, encoding="rgb8")
        header = Header()
        header.stamp = rospy.Time.now()
        msg.header = header
        pub.publish(msg)

    def plan_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False) -> Tuple[bool, Any, float, int]:
        assert pos.ndim == 1 and pos.shape[-1] == 3 and orn.ndim == 1 and orn.shape[-1] == 4 # Quaternion

        pose_goal = Pose()
        pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z  = pos
        if versor_comes_first:
            pose_goal.orientation.w, pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z = orn 
        else:
            pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w = orn 

        self.arm_group.clear_pose_targets()
        self.arm_group.set_pose_target(pose_goal)
        success, plan, planning_time, error_code = self.arm_group.plan()

        return success, plan, planning_time, error_code

    def move_to_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False) -> bool:
        success, plan, planning_time, error_code = self.plan_pose(pos=pos, orn=orn, versor_comes_first=versor_comes_first)
        if success is True:
            result: bool = self.arm_group.execute(plan_msg=plan, wait=True)
            rospy.loginfo(f"Execution result: {result}")
            self.arm_group.stop()
            return True
        else:
            rospy.loginfo(f"Plan failed. ErrorCode: {error_code}")
            self.arm_group.stop()
            return False
        


class EdfEnvRosInterface(EdfInterface):
    def __init__(self, env: UR5Env, monitor_refresh_rate: float = 0):
        self.ros_handle = UR5EnvRosHandle(env=env, monitor_refresh_rate=monitor_refresh_rate)

    def observe_scene(self, obs_type: str ='pointcloud', update: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], List[CamData]]:
        if obs_type == 'pointcloud':
            if update:
                self.ros_handle.update_scene_pc_msg()
            return self.ros_handle.scene_points, self.ros_handle.scene_colors
        elif obs_type == 'image':
            return self.ros_handle.env.observe_scene()
        else:
            raise ValueError("Wrong observation type is given.")

    def observe_ee(self, obs_type: str ='pointcloud', update: bool = True):
        raise NotImplementedError
        
    def pick(self, poses: np.ndarray) -> List[bool]:
        assert poses.ndim == 2 and poses.shape[-1] == 7 # [[qw,qx,qy,qz,x,y,z], ...]

        results = []
        for pose in poses:
            result_ = self.ros_handle.move_to_pose(pos=pose[4:], orn=pose[:4], versor_comes_first=True)
            results.append(result_)
            if result_ is True:
                break
        return results

    def place(self, poses):
        raise NotImplementedError