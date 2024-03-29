import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any, Dict

import numpy as np

import rospy
import actionlib
import tf2_ros
from ros_numpy.point_cloud2 import array_to_pointcloud2
from ros_numpy.image import numpy_to_image

from sensor_msgs.msg import JointState, PointCloud2, Image
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, Pose
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse, Trigger, TriggerRequest, TriggerResponse
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

from edf_env.env import UR5Env
from edf_env.pc_utils import encode_pc
from edf_env.utils import CamData









class UR5EnvRos():
    def __init__(self, env: UR5Env, monitor_refresh_rate: float = 0, reset_kwargs: Dict = {}):
        self.env_seed = None
        self.env: UR5Env = env
        self.env_ready = True
        self.reset_kwargs: Dict = reset_kwargs

        # self.current_traj: Optional[JointTrajectory] = None
        self.scene_pc_msg: Optional[PointCloud2] = None 
        self.eef_pc_msg: Optional[PointCloud2] = None 

        
        self.joint_pub = rospy.Publisher('joint_states', JointState, latch=False, queue_size=1)
        self.scene_pc_pub = rospy.Publisher('scene_pointcloud', PointCloud2, latch=False, queue_size=1)
        self.eef_pc_pub = rospy.Publisher('eef_pointcloud', PointCloud2, latch=False, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.arm_ctrl_AS_name = 'arm_controller/follow_joint_trajectory'
        self.arm_ctrl_AS = actionlib.SimpleActionServer(self.arm_ctrl_AS_name, FollowJointTrajectoryAction,
                                                              execute_cb = self.execute_cb, auto_start = False)                                                           
        
        self.monitor_refresh_rate = monitor_refresh_rate
        self.monitor_img_pubs = []
        for i in range(len(self.env.monitor_cam_configs)):
            self.monitor_img_pubs.append(rospy.Publisher(f"monitor_img_{i}", Image, latch=False, queue_size=1))
        # self.update_scene_pc_server = rospy.Service('update_scene_pointcloud', UpdatePointCloud, self.update_scene_pc_srv_callback)
        self.update_scene_pc_server = rospy.Service('update_scene_pointcloud', Empty, self.update_scene_pc_srv_callback)
        self.update_eef_pc_server = rospy.Service('update_eef_pointcloud', Empty, self.update_eef_pc_srv_callback)
        self.reset_env = rospy.Service('reset_env', Empty, self.reset_env_srv_callback)
        # self.attach_srv = rospy.Service('env_attach_srv', Trigger, self.attach_srv_callback)
        # self.detach_srv = rospy.Service('env_detach_srv', Trigger, self.detach_srv_callback)
        self.grasp_srv = rospy.Service('grasp_srv', Trigger, self.grasp_srv_callback)
        self.release_srv = rospy.Service('release_srv', Trigger, self.release_srv_callback)
        self.mobile_base_srv = rospy.Service('mobile_base_srv', GetPlan, self.mobile_base_srv_callback)

        rospy.init_node('edf_env', anonymous=True, log_level=rospy.INFO)
        self.arm_ctrl_AS.start()


        self.threads=[]
        self.threads.append(threading.Thread(name='jointpub_thread', target=self.jointpub_thread))
        self.threads.append(threading.Thread(name='tfpub_thread', target=self.tfpub_thread))
        self.threads.append(threading.Thread(name='scene_pcpub_thread', target=self.scene_pcpub_thread))
        self.threads.append(threading.Thread(name='eef_pcpub_thread', target=self.eef_pcpub_thread))
        if self.monitor_refresh_rate > 0:
            self.threads.append(threading.Thread(name='monitor_imgpub_thread', target=self.monitor_imgpub_thread))


        self.update_scene_pc_msg()
        for thread in self.threads:
            thread.start()


    def close(self):
        self.env.close()
        for thread in self.threads:
            thread.terminate()

    def monitor_imgpub_thread(self):
        rate = rospy.Rate(self.monitor_refresh_rate)
        while not rospy.is_shutdown():
            imgs = [cam_dat['color'] for cam_dat in self.env.observe_monitor_img(return_seg=False, color_encoding='uint8')]
            for img, pub in zip(imgs, self.monitor_img_pubs):
                if self.env_ready:
                    self.publish_image(img=img, pub=pub)
            rate.sleep()

    def jointpub_thread(self):
        # TODO:  https://stackoverflow.com/questions/50907224/not-able-to-terminate-the-process-in-multiprocessing-python-linux
        rate = rospy.Rate(20) # 10hz
        while not rospy.is_shutdown():
            if self.env_ready:
                self.publish_joint_info()
            rate.sleep()

    def tfpub_thread(self):
        rate = rospy.Rate(20) 
        while not rospy.is_shutdown():
            if self.env_ready:
                self.publish_base_link_tf()
                self.publish_scene_tf()
            rate.sleep()

    def scene_pcpub_thread(self):
        rate = rospy.Rate(2) 
        while not rospy.is_shutdown():
            if self.env_ready:
                self.publish_scene_pc()
            rate.sleep()

    def eef_pcpub_thread(self):
        rate = rospy.Rate(2) 
        while not rospy.is_shutdown():
            if self.env_ready:
                self.publish_eef_pc()
            rate.sleep()

    def publish_joint_info(self):
        state_dict = self.env.get_joint_state_dict()

        header = Header()
        header.stamp = rospy.Time.now()

        msg = JointState()
        msg.header = header
        for k,v in state_dict.items():
            msg.name.append(k)
            msg.position.append(v[0])
            msg.velocity.append(v[1])

        self.joint_pub.publish(msg)

    def execute_cb(self, goal):
        assert self.env_ready
        
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
        frame_id = self.env.world_frame_name

        points, colors = self.env.observe_scene_pc()
        pc = encode_pc(points=points, colors=colors)
        self.scene_pc_msg: PointCloud2 = array_to_pointcloud2(cloud_arr = pc, stamp=stamp, frame_id=frame_id)

    def update_eef_pc_msg(self):
        stamp = rospy.Time.now()
        frame_id = self.env.end_effector_link_name

        points, colors = self.env.observe_eef_pc()
        pc = encode_pc(points=points, colors=colors)
        self.eef_pc_msg: PointCloud2 = array_to_pointcloud2(cloud_arr = pc, stamp=stamp, frame_id=frame_id)

    def publish_scene_pc(self):
        if self.scene_pc_msg is None:
            self.update_scene_pc_msg()
        stamp = rospy.Time.now()
        self.scene_pc_msg.header.stamp = stamp
        self.scene_pc_pub.publish(self.scene_pc_msg)

    def publish_eef_pc(self):
        if self.eef_pc_msg is None:
            self.update_eef_pc_msg()
        stamp = rospy.Time.now()
        self.eef_pc_msg.header.stamp = stamp
        self.eef_pc_pub.publish(self.eef_pc_msg)

    def publish_base_link_tf(self):
        pos, orn = self.env.get_base_pose()
        
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.env.world_frame_name
        t.child_frame_id = self.env.base_frame_name
        t.transform.translation.x,  t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = orn

        self.tf_broadcaster.sendTransform(t)

    def publish_scene_tf(self):
        pos, orn = self.env.get_scene_pose()
        
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.env.world_frame_name
        t.child_frame_id = self.env.scene_frame_name
        t.transform.translation.x,  t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = orn

        self.tf_broadcaster.sendTransform(t)

    def publish_image(self, img: np.ndarray, pub: rospy.Publisher):
        msg = numpy_to_image(arr=img, encoding="rgb8")
        header = Header()
        header.stamp = rospy.Time.now()
        msg.header = header
        pub.publish(msg)

    # def update_scene_pc_srv_callback(self, request: UpdatePointCloudRequest) -> UpdatePointCloudResponse:
    #     self.update_scene_pc_msg()
    #     self.publish_scene_pc()

    #     result = UpdatePointCloudResponse.SUCCESS
    #     response = UpdatePointCloudResponse()
    #     response.result = result

    #     return response
    def update_scene_pc_srv_callback(self, request: EmptyRequest) -> EmptyResponse:
        if self.env_ready:
            self.update_scene_pc_msg()
            self.publish_scene_pc()
            time.sleep(0.1)
        else:
            pass

        return EmptyResponse()
    
    def update_eef_pc_srv_callback(self, request: EmptyRequest) -> EmptyResponse:
        if self.env_ready:
            self.update_eef_pc_msg()
            self.publish_eef_pc()
            time.sleep(0.1)
        else:
            pass

        return EmptyResponse()
    
    def reset_env_srv_callback(self, request: EmptyRequest) -> EmptyResponse:
        self.env_ready = False
        self.env.reset(seed = self.env_seed, **self.reset_kwargs)
        self.eef_pc_msg = None
        self.scene_pc_msg = None
        self.env_ready = True
        
        self.update_scene_pc_srv_callback(request=EmptyRequest())

        return EmptyResponse()
    
    # def attach_srv_callback(self, request: TriggerRequest) -> TriggerResponse:
    #     result = self.env.attach(item_id=self.env.target_obj_id)
    #     if result == 'ALREADY_IN_GRASP' or result == 'SUCCESS':
    #         success = True
    #     else:
    #         success = False

    #     return TriggerResponse(success=success, message=result)
    
    # def detach_srv_callback(self, request: TriggerRequest) -> TriggerResponse:
    #     result = self.env.detach()
    #     if result == 'NO_ATTACHED_OBJ' or result == 'SUCCESS':
    #         success = True
    #     else:
    #         success = False

    #     return TriggerResponse(success=success, message=result)
    
    def grasp_srv_callback(self, request: TriggerRequest) -> TriggerResponse:
        success = self.env.grasp()
        if success:
            result = 'SUCCESS'
        else:
            result = 'FAIL_UNKNOWN'

        return TriggerResponse(success=success, message=result)
    
    def release_srv_callback(self, request: TriggerRequest) -> TriggerResponse:
        success = self.env.release()
        if success:
            result = 'SUCCESS'
        else:
            result = 'FAIL_UNKNOWN'

        return TriggerResponse(success=success, message=result)
    
    def mobile_base_srv_callback(self, request: GetPlanRequest) -> GetPlanResponse:
        pose: Pose = request.goal.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        orn = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.env.move_robot_base(target_pos=pos[:2])

        return GetPlanResponse()
