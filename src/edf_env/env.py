import time
import os, sys
from typing import Union, Optional, Type, TypedDict, Any, List, Dict, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import yaml

import edf_env
from edf_env.utils import HideOutput, observe_cams, CamData, CamConfig, load_yaml, load_joints_info
from edf_env.pybullet_pc_utils import pb_cams_to_pc
from edf_env.pc_utils import pcd_from_numpy






class BulletEnv():
    def __init__(self, use_gui: bool = True, sim_freq: float = 1000):
        """A template class of pybullet environment.

        Args:
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).

        """

        with HideOutput():                     # To prevent verbose Pybullet details from being printed.
            self.physicsClientId: int = p.connect(p.GUI if use_gui else p.DIRECT )
        if use_gui:
            p.configureDebugVisualizer(flag = p.COV_ENABLE_MOUSE_PICKING, enable = 0)
        self.sim_freq: float = sim_freq
        self.init_task()

    def init_task(self):
        """Initialize task environment."""

        p.setAdditionalSearchPath(pybullet_data.getDataPath())                                      # Search for default assets directory.
        p.setGravity(0,0,-10, physicsClientId = self.physicsClientId)
        self.plane_id: int = p.loadURDF("plane.urdf", physicsClientId = self.physicsClientId)       # Spawn plane.
        p.setTimeStep(1/self.sim_freq)

    def close(self):
        """Disconnect from the Pybullet server."""

        p.disconnect()

    def load_cam_config(self, cam_config_path: str) -> List[CamConfig]:
        """Loads list of pybullet camera configs from yaml file path."""
        cam_configs: List[CamConfig] = load_yaml(cam_config_path)
        return cam_configs

    def observe_cams(self, cam_configs: List[CamConfig], target_pos: Optional[np.ndarray] = None, return_seg: bool = False, color_encoding: str = "float") -> List[CamData]:
        """Observes multiple pybullet virtual camera data from list of camera configurations.
        If target_pos is specified, all the cameras will look at the same fixation point in the specificed target position.

        Returns:
            A list of camera data for cam_configs. See :func:`~edf_env.utils.get_pybullet_cam_data` for details as to individual camera config/data.

        Args:
            cam_config (List[CamConfig]): A typed dict that specifies camera configurations.
            target_pos (Optional[np.ndarray]): If target_pos is specified, all the cameras will look at the same fixation point in the specificed target position.

            Shape::

                - target_pos: (3,)
                
        """
        
        return observe_cams(cam_configs=cam_configs, target_pos=target_pos, return_seg = return_seg, color_encoding=color_encoding, physicsClientId=self.physicsClientId)
        
    def observe(self):
        raise NotImplementedError



class UR5Env(BulletEnv):
    def __init__(self, 
                 env_config_path: str = os.path.join(edf_env.ROOT_DIR, 'config/env_config.yaml'), 
                 scene_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/scene_camera_config.yaml'), 
                 grasp_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/grasp_camera_config.yaml'), 
                 monitor_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/monitor_camera_config.yaml'),
                 robot_path: str = os.path.join(edf_env.ROOT_DIR, 'robot/ridgeback_ur5/ridgeback_ur5_robotiq.urdf'), 
                 use_gui: bool = True, 
                 sim_freq: float = 1000):
        """Pybullet environment with UR5 Robot

        Args:
            env_config_path: Path to the pybullet environment configuration file.
            scene_cam_config_path: Path to the camera configuration file for scene observation.
            grasp_cam_config_path: Path to the camera configuration file for observing robot's hand (to see how the robot is gripping the object).
            robot_path: Path to the robot description file (in URDF format).
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).
            self.scene_cam_configs: A list of pybullet camera configurations for scene observation.
            self.grasp_cam_configs: A list of pybullet camera configurations for observing robot's hand (to see how the robot is gripping the object).

        """
        super().__init__(use_gui=use_gui, sim_freq=sim_freq)
        
        self.load_env_config(config_path=env_config_path)

        ############ Load robot ################################################
        self.robot_id: int = self.load_robot(urdf_path=robot_path)
        self.robot_joint_dict, self.robot_joint_type_dict, self.n_joints = load_joints_info(body_id=self.robot_id, physicsClientId=self.physicsClientId)
        self.init_robot_pose()

        # self.movable_joint_ids = []                    # Idx: Movable joint idx (0~5) || Val: Pybullet jointId (0~27)
        # for k,v in self.robot_joint_dict.items():
        #     if type(k) == int and self.robot_joint_type_dict[k] is not 'JOINT_FIXED':
        #         self.movable_joint_ids.append(k)
        # self.movable_joint_ids.sort()

        # self.movable_joint_idx_dict: Dict[str, int] = {} # Key: Joint name || Val: Movable joint idx (0~5)         **Note that Movable joint idx (0~5) is different from Pybullet jointId (0~27)
        # for n, id in enumerate(self.movable_joint_ids):
        #     self.movable_joint_idx_dict[self.robot_joint_dict[id]] = n
        
        ############ Load table ################################################
        if self.spawn_table:
            self.table_id = p.loadURDF(edf_env.ROOT_DIR + "/assets/table.urdf", basePosition=self.table_pos, baseOrientation=p.getQuaternionFromEuler(self.table_rpy), globalScaling=self.table_scale, physicsClientId = self.physicsClientId)

        ############ Load camera configurations ################################################
        if scene_cam_config_path is None:
            self.scene_cam_configs = None
        else:
            self.scene_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=scene_cam_config_path)

        if grasp_cam_config_path is None:
            self.grasp_cam_configs = None
        else:
            self.grasp_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=grasp_cam_config_path)

        if monitor_cam_config_path is None:
            self.monitor_cam_configs = None
        else:
            self.monitor_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=monitor_cam_config_path)
        #########################################################################################

        # self.disable_gripper_self_collision()
        self.set_gripper_constraint()

       
    def load_env_config(self, config_path: str):
        """Loads environment config from yaml file path."""
        config: Dict[str, Any] = load_yaml(config_path)

        robot_config: Dict[str, Any] = config['robot_config']
        self.robot_base_pose_init: Dict[str, Optional[np.ndarray]] = robot_config['robot_base_pose_init']
        if self.robot_base_pose_init['pos'] is None:
            self.robot_base_pose_init['pos'] = np.array([0.0, 0.0, 0.0])
        if self.robot_base_pose_init['orn'] is None:
            self.robot_base_pose_init['orn'] = np.array([0.0, 0.0, 0.0, 1.0])
        self.robot_joint_init = robot_config['robot_joint_init']

        table_config: Dict[str, Any] = config['table_config']
        if table_config['spawn'] == True:
            self.spawn_table = True
            self.table_scale = table_config['scale']
            self.table_pos = np.array(table_config['pos'])
            self.table_rpy = table_config['rpy']
            self.table_rel_center = np.array(table_config['center']) * self.table_scale
            self.table_center = self.table_pos + self.table_rel_center
        else:
            self.spawn_table = False

        scene_config: Dict[str, Any] = config['scene_config']
        self.scene_center: np.ndarray = np.array(scene_config['scene_center']) # [xc,yc,zc]: Shape: (3,)
        if self.spawn_table is True:
            self.scene_center = self.scene_center + self.table_center
        self.scene_ranges: np.ndarray = np.array(scene_config['scene_ranges']) # [[x_min, x_max], [y_min, y_max], [z_min, z_max]];  Shape: (3,2)
        if scene_config['relative_range'] is True:
            self.scene_ranges = self.scene_ranges + np.stack([self.scene_center, self.scene_center], axis=1)
        else:
            raise NotImplementedError
        self.scene_voxel_filter_size = scene_config['pc_voxel_filter_size']

    def load_robot(self, urdf_path: str) -> int:
        """Loads list of pybullet camera configs from yaml file path."""
        robot_id = p.loadURDF(fileName=urdf_path, physicsClientId=self.physicsClientId, basePosition = self.robot_base_pose_init['pos'], baseOrientation = self.robot_base_pose_init['orn'], useFixedBase = True)
        return robot_id

    def observe_monitor_img(self, target_pos: Optional[np.ndarray] = None, return_seg: bool = False, color_encoding: str = "float") -> List[CamData]:
        if target_pos is None:
            target_pos = self.scene_center
        return self.observe_cams(cam_configs=self.monitor_cam_configs, target_pos=target_pos, return_seg=return_seg, color_encoding=color_encoding)

    def observe_scene(self, stride: Union[np.ndarray, list, tuple] = (1,1), return_seg: bool = False, color_encoding: str = "float") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[CamData]]:
        """Get point cloud and camera observation data of the scene.

        Returns:
            1) point cloud 3d-coordinate
            2) point cloud color
            3) point cloud segmentation (optional)
            4) list of raw camera data (See See :func:`get_pybullet_cam_data` for details)

            Shape::

                - coord (np.ndarray): (N_points, 3)
                - color (np.ndarray): (N_points, 3)
                - seg (np.ndarray[int] or None): (N_points)

        Args:
            stride: (np.ndarray, list or tuple): Stride of sampling points from images. (1,1) means dense sampling.

            Shape::

                - stride: (2,) 

        """
        assert self.scene_cam_configs, "Scene camera configuration is not assigned."

        cam_data_list: List[CamData] = self.observe_cams(cam_configs=self.scene_cam_configs, target_pos=self.scene_center, return_seg=return_seg, color_encoding=color_encoding)
        pc_coord, pc_color, pc_seg = pb_cams_to_pc(cam_data_list=cam_data_list, ranges=self.scene_ranges, stride=stride)

        return pc_coord, pc_color, pc_seg, cam_data_list

    def observe_scene_pc(self, voxel_filter_size: Optional[float] = None, segmented: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """DOCSTRING TODO"""
        if voxel_filter_size is None:
            voxel_filter_size = self.scene_voxel_filter_size

        points, colors, pc_seg, cam_data_list = self.observe_scene(return_seg = segmented, color_encoding = "float")
        if segmented is True:
            raise NotImplementedError
        pcd = pcd_from_numpy(coord=points, color=colors, voxel_filter_size = voxel_filter_size)
        
        return np.asarray(pcd.points), np.asarray(pcd.colors)

    def _get_joint_states_from_id_list(self, joint_ids: List[int]) -> Tuple[List[float], List[float]]:
        """Docstring TODO"""
        states = p.getJointStates(bodyUniqueId = self.robot_id, jointIndices=joint_ids)
        pos = [s[0] for s in states] # Shape: (N_joints,)
        vel = [s[1] for s in states] # Shape: (N_joints,)
        # force = np.stack([np.array(s[2]) for s in states], axis=-2) # list of [Fx, Fy, Fz, Mx, My, Mz]  =>  Shape: (N_joints, 6)
        # applied_torque = np.array([s[3] for s in states])           # Shape: (N_joints,)

        return pos, vel

    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        """Docstring TODO"""
        return self._get_joint_states_from_id_list(joint_ids=list(range(self.n_joints)))

    def step(self):
        # self.gripper_mimic_constraint()
        p.stepSimulation(physicsClientId = self.physicsClientId)

    def position_control(self, control_joint_IDs, target_pos):
        for id, pos in zip(control_joint_IDs, target_pos):
            if id == self.robot_joint_dict['finger_joint']:
                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['left_inner_knuckle_joint'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = pos,
                                        physicsClientId=self.physicsClientId) 
                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['right_inner_knuckle_joint'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = -pos,
                                        physicsClientId=self.physicsClientId) 
            else:
                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = id,
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = pos,
                                        physicsClientId=self.physicsClientId) 

    def control_target_joint_states(self, target_pos: List[float], target_vel: List[float], target_duration: float, target_joint_names: List[str]) -> bool:
        """Docstring TODO"""
        control_joint_IDs = []
        for name in target_joint_names:
            control_joint_IDs.append(self.robot_joint_dict[name])

        target_duration = 0.8 * target_duration
        target_steps = int(target_duration * self.sim_freq)
        target_pos, target_vel = np.array(target_pos), np.array(target_vel)                                           # Shape: (N_target_joints, 3), (N_target_joints, 3)

        current_pos, current_vel = self.get_joint_states()
        current_pos, current_vel = np.array(current_pos)[control_joint_IDs], np.array(current_vel)[control_joint_IDs] # Shape: (N_target_joints, 3), (N_target_joints, 3)
        

        for step in range(target_steps):
            r = step / target_steps
            pos = r * (target_pos - current_pos) + current_pos
            vel = r * (target_vel - current_vel) + current_vel

            force = None
            if force is None:
                self.position_control(control_joint_IDs=control_joint_IDs, target_pos=pos)
            else:
                raise NotImplementedError
            self.step()

        return True

    def get_base_pose(self):
        pos, orn = p.getBasePositionAndOrientation(bodyUniqueId = self.robot_id, physicsClientId = self.physicsClientId)
        return pos, orn

    def init_robot_pose(self):
        for config in self.robot_joint_init:
            name = config['name']
            value = config['value']
            p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict[name], targetValue = value)

    def set_gripper_constraint(self):
        constraints = []
        for _ in range(3):
            c = p.createConstraint(parentBodyUniqueId = self.robot_id,
                                parentLinkIndex = self.robot_joint_dict['left_outer_finger_joint'],
                                childBodyUniqueId = self.robot_id,
                                childLinkIndex = self.robot_joint_dict['left_inner_finger_joint'],
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[0, 0, 0],
                                parentFramePosition=[0.001, 0, 0.0265],
                                childFramePosition=[0.012, 0.0, -0.013],
                                physicsClientId=self.physicsClientId)
            p.changeConstraint(c, erp=0.1, maxForce=1000, physicsClientId=self.physicsClientId)
            constraints.append(c)

            c = p.createConstraint(parentBodyUniqueId = self.robot_id,
                                parentLinkIndex = self.robot_joint_dict['left_inner_knuckle_joint'],
                                childBodyUniqueId = self.robot_id,
                                childLinkIndex = self.robot_joint_dict['left_inner_finger_joint'],
                                jointType=p.JOINT_GEAR,
                                jointAxis=[0, 1, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0],
                                physicsClientId=self.physicsClientId)
            p.changeConstraint(c, gearRatio=1, maxForce=100, erp=2, physicsClientId=self.physicsClientId)
            constraints.append(c)


            c = p.createConstraint(parentBodyUniqueId = self.robot_id,
                                parentLinkIndex = self.robot_joint_dict['right_outer_finger_joint'],
                                childBodyUniqueId = self.robot_id,
                                childLinkIndex = self.robot_joint_dict['right_inner_finger_joint'],
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[0, 0, 0],
                                parentFramePosition=[0.001, 0, 0.0265],
                                childFramePosition=[0.012, 0.0, -0.013],
                                physicsClientId=self.physicsClientId)
            p.changeConstraint(c, erp=0.1, maxForce=1000, physicsClientId=self.physicsClientId)
            constraints.append(c)

            c = p.createConstraint(parentBodyUniqueId = self.robot_id,
                                parentLinkIndex = self.robot_joint_dict['right_inner_knuckle_joint'],
                                childBodyUniqueId = self.robot_id,
                                childLinkIndex = self.robot_joint_dict['right_inner_finger_joint'],
                                jointType=p.JOINT_GEAR,
                                jointAxis=[0, 1, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0],
                                physicsClientId=self.physicsClientId)
            p.changeConstraint(c, gearRatio=1, maxForce=100, erp=2, physicsClientId=self.physicsClientId)
            constraints.append(c)

        return constraints

    # def disable_gripper_self_collision(self):
    #     p.setCollisionFilterPair(self.robot_id, self.robot_id, self.robot_joint_dict['left_outer_finger_joint'], self.robot_joint_dict['left_inner_finger_joint'], 0, self.physicsClientId)
    #     p.setCollisionFilterPair(self.robot_id, self.robot_id, self.robot_joint_dict['right_outer_finger_joint'], self.robot_joint_dict['right_inner_finger_joint'], 0, self.physicsClientId)

    # def gripper_mimic_constraint(self, max_torque: float = 500):
    #     target_val = p.getJointState(bodyUniqueId = self.robot_id, jointIndex = self.robot_joint_dict['finger_joint'], physicsClientId = self.physicsClientId)[0]
        

    #     p.setJointMotorControl2(bodyIndex = self.robot_id,
    #                             jointIndex = self.robot_joint_dict['left_inner_knuckle_joint'],
    #                             controlMode = p.POSITION_CONTROL,
    #                             targetPosition = target_val,
    #                             physicsClientId = self.physicsClientId,
    #                             force = max_torque
    #                             )
    #     p.setJointMotorControl2(bodyIndex = self.robot_id,
    #                             jointIndex = self.robot_joint_dict['left_inner_finger_joint'],
    #                             controlMode = p.POSITION_CONTROL,
    #                             targetPosition = -1 * target_val,
    #                             physicsClientId = self.physicsClientId,
    #                             force = max_torque
    #                             )
    #     p.setJointMotorControl2(bodyIndex = self.robot_id,
    #                             jointIndex = self.robot_joint_dict['right_outer_knuckle_joint'],
    #                             controlMode = p.POSITION_CONTROL,
    #                             targetPosition = -1 * target_val,
    #                             physicsClientId = self.physicsClientId,
    #                             force = max_torque
    #                             )
    #     p.setJointMotorControl2(bodyIndex = self.robot_id,
    #                             jointIndex = self.robot_joint_dict['right_inner_knuckle_joint'],
    #                             controlMode = p.POSITION_CONTROL,
    #                             targetPosition = -1 * target_val,
    #                             physicsClientId = self.physicsClientId,
    #                             force = max_torque
    #                             )
    #     p.setJointMotorControl2(bodyIndex = self.robot_id,
    #                             jointIndex = self.robot_joint_dict['right_inner_finger_joint'],
    #                             controlMode = p.POSITION_CONTROL,
    #                             targetPosition = 1 * target_val,
    #                             physicsClientId = self.physicsClientId,
    #                             force = max_torque
    #                             )

    # def control_gripper(self, target_val: float, duration: float) -> bool:
    #     return self.control_target_joint_states(self, target_pos = [target_val], target_vel= [0], target_duration = duration, target_joint_names = ['finger_joint'])
    








class MugEnv(UR5Env):
    def __init__(self, 
                 env_config_path: str = os.path.join(edf_env.ROOT_DIR, 'config/env_config.yaml'), 
                 scene_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/scene_camera_config.yaml'), 
                 grasp_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/grasp_camera_config.yaml'), 
                 monitor_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/monitor_camera_config.yaml'),
                 robot_path: str = os.path.join(edf_env.ROOT_DIR, 'robot/ridgeback_ur5/ridgeback_ur5_robotiq.urdf'), 
                 use_gui: bool = True, 
                 sim_freq: float = 1000):
        """Pybullet environment with UR5 Robot and a mug with hanger

        Args:
            env_config_path: Path to the pybullet environment configuration file.
            scene_cam_config_path: Path to the camera configuration file for scene observation.
            grasp_cam_config_path: Path to the camera configuration file for observing robot's hand (to see how the robot is gripping the object).
            robot_path: Path to the robot description file (in URDF format).
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).
            self.scene_cam_configs: A list of pybullet camera configurations for scene observation.
            self.grasp_cam_configs: A list of pybullet camera configurations for observing robot's hand (to see how the robot is gripping the object).

        """

        super().__init__(env_config_path=env_config_path, scene_cam_config_path=scene_cam_config_path, grasp_cam_config_path=grasp_cam_config_path, monitor_cam_config_path=monitor_cam_config_path, robot_path=robot_path, use_gui=use_gui, sim_freq=sim_freq)
        self.mug_id = self.spawn_mug('train_0', pos = self.scene_center + np.array([0, 0, 0.1]), scale = 1.5)

        for _ in range(1000):
            self.step()

    def spawn_mug(self, mug_name: str, pos: np.ndarray, orn: Optional[np.ndarray] = None, scale: float = 1.0) -> int:
        if orn is None:
            return p.loadURDF(os.path.join(edf_env.ROOT_DIR, f"assets/mug/{mug_name}.urdf"), basePosition=pos, globalScaling=scale, physicsClientId = self.physicsClientId)
        
        raise NotImplementedError