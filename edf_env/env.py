import time
import os, sys
from typing import Union, Optional, Type, TypedDict, Any, List, Dict, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import yaml

import edf_env
from edf_env.utils import HideOutput, observe_cams, CamData, CamConfig, load_yaml, load_joints_info, load_links_info
from edf_env.pybullet_pc_utils import pb_cams_to_pc
from edf_env.pc_utils import pcd_from_numpy






class BulletEnv():
    def __init__(self, use_gui: bool = True, sim_freq: float = 1000, debug: bool = False):
        """A template class of pybullet environment.

        Args:
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).

        """
        self.use_gui = use_gui
        with HideOutput():                     # To prevent verbose Pybullet details from being printed.
            if use_gui:
                self.physicsClientId: int = p.connect(p.GUI, options='--width=800 --height=1000')
            else:
                self.physicsClientId: int = p.connect(p.DIRECT)
        if use_gui:
            if not debug:
                p.configureDebugVisualizer(flag = p.COV_ENABLE_MOUSE_PICKING, enable = 0)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.sim_freq: float = sim_freq

        # https://github.com/bulletphysics/bullet3/issues/1936
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/switchConstraintSolver.py
        p.setPhysicsEngineParameter(solverResidualThreshold=0.001, numSolverIterations=200, physicsClientId=self.physicsClientId)

        if type(self) == BulletEnv:
            self.reset()

    def reset(self, seed: Optional[int] = None) -> bool:
        """Reset task environment."""
        self.rng: np.random._generator.Generator = np.random.default_rng(seed=seed)
        p.resetSimulation(physicsClientId=self.physicsClientId)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())                                      # Search for default assets directory.
        p.setGravity(0,0,-10, physicsClientId = self.physicsClientId)
        self.plane_id: int = p.loadURDF("plane.urdf", physicsClientId = self.physicsClientId)       # Spawn plane.
        p.setTimeStep(1/self.sim_freq)

        return True

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
                 eef_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/eef_camera_config.yaml'), 
                 monitor_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/monitor_camera_config.yaml'),
                 robot_path: str = os.path.join(edf_env.ROOT_DIR, 'robot/ridgeback_ur5/ridgeback_ur5_robotiq.urdf'), 
                 use_gui: bool = True, 
                 sim_freq: float = 1000,
                 debug: bool = False):
        """Pybullet environment with UR5 Robot

        Args:
            env_config_path: Path to the pybullet environment configuration file.
            scene_cam_config_path: Path to the camera configuration file for scene observation.
            eef_cam_config_path: Path to the camera configuration file for observing robot's hand (to see how the robot is gripping the object).
            robot_path: Path to the robot description file (in URDF format).
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).
            self.scene_cam_configs: A list of pybullet camera configurations for scene observation.
            self.eef_cam_configs: A list of pybullet camera configurations for observing robot's hand (to see how the robot is gripping the object).

        """

        self.env_config_path: str = env_config_path
        self.scene_cam_config_path: Optional[str] = scene_cam_config_path
        self.eef_cam_config_path: Optional[str] = eef_cam_config_path
        self.monitor_cam_config_path: Optional[str] = monitor_cam_config_path
        self.robot_path: str = robot_path
        self.target_obj_id: Optional[int] = None

        super().__init__(use_gui=use_gui, sim_freq=sim_freq, debug=debug)
        if type(self) == UR5Env:
            self.reset()

    def reset(self, seed: Optional[int] = None) -> bool:
        super().reset(seed=seed)

        self.load_env_config(config_path=self.env_config_path)

        ############ Load robot ################################################
        self.robot_id: int = self.load_robot(urdf_path=self.robot_path)
        self.robot_joint_dict, self.robot_joint_type_dict, self.n_joints = load_joints_info(body_id=self.robot_id, physicsClientId=self.physicsClientId)
        self.robot_links_dict, self.n_robot_links = load_links_info(body_id=self.robot_id, physicsClientId=self.physicsClientId)
        self.end_effector_link_id = self.robot_links_dict[self.end_effector_link_name]
        self._fake_joints: set = set(["left_finger_z_joint_fake", "right_finger_z_joint_fake", "left_inner_finger_joint_fake", "right_inner_finger_joint_fake"])
        self.init_robot_pose(init_gripper=True)

        # self.movable_joint_ids = []                    # Idx: Movable joint idx (0~5) || Val: Pybullet jointId (0~27)
        # for k,v in self.robot_joint_dict.items():
        #     if type(k) == int and self.robot_joint_type_dict[k] is not 'JOINT_FIXED':
        #         self.movable_joint_ids.append(k)
        # self.movable_joint_ids.sort()

        # self.movable_joint_idx_dict: Dict[str, int] = {} # Key: Joint name || Val: Movable joint idx (0~5)         **Note that Movable joint idx (0~5) is different from Pybullet jointId (0~27)
        # for n, id in enumerate(self.movable_joint_ids):
        #     self.movable_joint_idx_dict[self.robot_joint_dict[id]] = n
        
        # self.disable_gripper_self_collision()
        self.set_gripper_constraint()
        self.grasp_constraint: List[int] = []
        self.grasped_item: Optional[int] = None
        self.detach()


        ############ Load table ################################################
        if self.spawn_table:
            self.table_id = p.loadURDF(edf_env.ROOT_DIR + "/assets/table.urdf", basePosition=self.table_pos, baseOrientation=p.getQuaternionFromEuler(self.table_rpy), globalScaling=self.table_scale, physicsClientId = self.physicsClientId)


        ############ Load camera configurations ################################################
        if self.scene_cam_config_path is None:
            self.scene_cam_configs = None
        else:
            self.scene_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=self.scene_cam_config_path)

        if self.eef_cam_config_path is None:
            self.eef_cam_configs = None
        else:
            self.eef_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=self.eef_cam_config_path)

        if self.monitor_cam_config_path is None:
            self.monitor_cam_configs = None
        else:
            self.monitor_cam_configs: List[CamConfig] = self.load_cam_config(cam_config_path=self.monitor_cam_config_path)
            debug_config = self.monitor_cam_configs[0]
            p.resetDebugVisualizerCamera(cameraDistance = debug_config['distance'], cameraYaw = debug_config['ypr'][0], cameraPitch = debug_config['ypr'][1], cameraTargetPosition = self.scene_center, physicsClientId=self.physicsClientId)

        p.changeDynamics(self.robot_id, self.robot_links_dict[self.lfinger_link_name], lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        p.changeDynamics(self.robot_id, self.robot_links_dict[self.rfinger_link_name], lateralFriction = 3., rollingFriction=3., spinningFriction=3.)
        #########################################################################################

        return True
       
    def load_env_config(self, config_path: str):
        """Loads environment config from yaml file path."""
        config: Dict[str, Any] = load_yaml(config_path)

        name_config: Dict[str, Any] = config['name_config']
        self.world_frame_name = name_config['world_frame_name']
        self.scene_frame_name = name_config['scene_frame_name']
        self.base_frame_name = name_config['base_frame_name']

        robot_config: Dict[str, Any] = config['robot_config']
        self.robot_base_pose_init: Dict[str, Optional[np.ndarray]] = robot_config['robot_base_pose_init']
        if self.robot_base_pose_init['pos'] is None:
            self.robot_base_pose_init['pos'] = np.array([0.0, 0.0, 0.0])
        if self.robot_base_pose_init['orn'] is None:
            self.robot_base_pose_init['orn'] = np.array([0.0, 0.0, 0.0, 1.0])
        self.robot_joint_init = robot_config['robot_joint_init']
        self.end_effector_link_name = robot_config['end_effector_link_name']
        self.lfinger_link_name = robot_config['lfinger_link_name']
        self.rfinger_link_name = robot_config['rfinger_link_name']
        self.gripper_max_force = robot_config['gripper_max_force']


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

        eef_config: Dict[str, Any] = config['eef_config']
        self.eef_ranges: np.ndarray = np.array(eef_config['eef_ranges']) # [[x_min, x_max], [y_min, y_max], [z_min, z_max]];  Shape: (3,2)
        self.eef_voxel_filter_size = eef_config['pc_voxel_filter_size']
        self.eef_grasp_point: np.ndarray = np.array(eef_config['grasp_point'])


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
    
    def observe_eef(self, stride: Union[np.ndarray, list, tuple] = (1,1), return_seg: bool = False, color_encoding: str = "float") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[CamData]]:
        """Get point cloud and camera observation data of the end_effector.

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
        assert self.eef_cam_configs, "Grasp camera configuration is not assigned."

        eef_pos, eef_orn = self.get_link_pose(link_id=self.end_effector_link_id)
        eef_pos, eef_orn = np.array(eef_pos), np.array(eef_orn)
        target_pos = Rotation.from_quat(eef_orn).apply(self.eef_grasp_point) + eef_pos
        cam_data_list: List[CamData] = self.observe_cams(cam_configs=self.eef_cam_configs, target_pos=target_pos, return_seg=return_seg, color_encoding=color_encoding)
        pc_coord, pc_color, pc_seg = pb_cams_to_pc(cam_data_list=cam_data_list, ranges=self.eef_ranges, stride=stride, frame=(eef_pos, eef_orn))

        return pc_coord, pc_color, pc_seg, cam_data_list

    def observe_eef_pc(self, voxel_filter_size: Optional[float] = None, segmented: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """DOCSTRING TODO"""
        if voxel_filter_size is None:
            voxel_filter_size = self.eef_voxel_filter_size

        points, colors, pc_seg, cam_data_list = self.observe_eef(return_seg = segmented, color_encoding = "float")
        if segmented is True:
            raise NotImplementedError
        pcd = pcd_from_numpy(coord=points, color=colors, voxel_filter_size = voxel_filter_size)
        
        return np.asarray(pcd.points), np.asarray(pcd.colors)

    def _get_joint_state(self, joint_id) -> Tuple[float, float]:
        name = self.robot_joint_dict[joint_id]
        if name == 'finger_joint':
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['left_inner_finger_joint_fake'])
            x_pos, x_vel = jstate[0], jstate[1]
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['left_finger_z_joint_fake'])
            z_pos, z_vel = jstate[0], jstate[1]
            pos, vel = self._pris_to_rev_finger(z_disp=z_pos, x_disp=x_pos, right_finger=False), self._pris_to_rev_finger(z_disp=z_vel, x_disp=x_vel, right_finger=False)
            pos, vel = -pos, -vel
        if name == 'left_inner_finger_joint':
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['left_inner_finger_joint_fake'])
            x_pos, x_vel = jstate[0], jstate[1]
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['left_finger_z_joint_fake'])
            z_pos, z_vel = jstate[0], jstate[1]
            pos, vel = self._pris_to_rev_finger(z_disp=z_pos, x_disp=x_pos, right_finger=False), self._pris_to_rev_finger(z_disp=z_vel, x_disp=x_vel, right_finger=False)
        elif name == 'right_inner_finger_joint':
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['right_inner_finger_joint_fake'])
            x_pos, x_vel = jstate[0], jstate[1]
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=self.robot_joint_dict['right_finger_z_joint_fake'])
            z_pos, z_vel = jstate[0], jstate[1]
            pos, vel = self._pris_to_rev_finger(z_disp=z_pos, x_disp=x_pos, right_finger=True), self._pris_to_rev_finger(z_disp=z_vel, x_disp=x_vel, right_finger=False)
        else:
            jstate = p.getJointState(bodyUniqueId = self.robot_id, jointIndex=joint_id)
            pos, vel = jstate[0], jstate[1]

        return pos, vel

    def _get_joint_states_from_id_list(self, joint_ids: List[int]) -> Tuple[List[float], List[float]]:
        """Docstring TODO"""
        pos_list, vel_list = [], []
        for id in joint_ids:
            pos, vel = self._get_joint_state(joint_id=id)
            pos_list.append(pos)
            vel_list.append(vel)

        return pos_list, vel_list

    def _get_joint_states(self) -> Tuple[List[float], List[float]]:
        """Docstring TODO"""
        return self._get_joint_states_from_id_list(joint_ids=list(range(self.n_joints)))

    def get_joint_state_dict(self) -> Dict[str, Tuple[float, float]]:
        state_dict = {}
        for k,v in self.robot_joint_dict.items():
            if type(k) == str:
                if k in self._fake_joints:
                    pass
                else:
                    state_dict[k] = self._get_joint_state(v)
        return state_dict

    def step(self):
        self.gripper_mimic_constraint()
        p.stepSimulation(physicsClientId = self.physicsClientId)

    def _rev_to_pris_finger(self, angle, right_finger = False):
        if right_finger:
            x_base = -1 * self.x_base
            theta = -self.angle_base + angle
        else:
            x_base = self.x_base
            theta = self.angle_base + angle
        
        z_disp = self.knuckle_len * np.cos(theta) + self.z_base
        x_disp = self.knuckle_len * np.sin(theta) + x_base

        return z_disp, x_disp
    
    def _pris_to_rev_finger(self, z_disp, x_disp, right_finger = False):
        if right_finger:
            x_base = -1 * self.x_base
        else:
            x_base = self.x_base

        theta = np.arctan2(x_disp-x_base, z_disp-self.z_base)

        if right_finger:
            angle = theta + self.angle_base
        else:
            angle = theta - self.angle_base
        
        return angle


    def _position_control(self, control_joint_IDs, target_pos):
        for id, pos in zip(control_joint_IDs, target_pos):
            if id == self.robot_joint_dict['finger_joint']:
                self._position_control(control_joint_IDs=[self.robot_joint_dict['left_inner_finger_joint']], target_pos=[-pos])
                self._position_control(control_joint_IDs=[self.robot_joint_dict['right_inner_finger_joint']], target_pos=[pos])

            elif id == self.robot_joint_dict['left_inner_finger_joint']:
                z_disp, x_disp = self._rev_to_pris_finger(angle=pos, right_finger=False)

                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['left_inner_finger_joint_fake'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = x_disp,
                                        force = self.gripper_max_force,
                                        physicsClientId=self.physicsClientId) 
                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['left_finger_z_joint_fake'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = z_disp,
                                        force = self.gripper_max_force,
                                        physicsClientId=self.physicsClientId) 
                
            elif id == self.robot_joint_dict['right_inner_finger_joint']:
                z_disp, x_disp = self._rev_to_pris_finger(angle=pos, right_finger=True)

                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['right_inner_finger_joint_fake'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = x_disp,
                                        force = self.gripper_max_force,
                                        physicsClientId=self.physicsClientId) 
                p.setJointMotorControl2(bodyIndex = self.robot_id,
                                        jointIndex = self.robot_joint_dict['right_finger_z_joint_fake'],
                                        controlMode = p.POSITION_CONTROL,
                                        targetPosition = z_disp,
                                        force = self.gripper_max_force,
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

        current_pos, current_vel = self._get_joint_states()
        current_pos, current_vel = np.array(current_pos)[control_joint_IDs], np.array(current_vel)[control_joint_IDs] # Shape: (N_target_joints, 3), (N_target_joints, 3)

        for step in range(target_steps):
            r = step / target_steps
            pos = r * (target_pos - current_pos) + current_pos
            vel = r * (target_vel - current_vel) + current_vel

            force = None
            if force is None:
                self._position_control(control_joint_IDs=control_joint_IDs, target_pos=pos)
            else:
                raise NotImplementedError
            self.step()

        return True

    def get_base_pose(self):
        pos, orn = p.getBasePositionAndOrientation(bodyUniqueId = self.robot_id, physicsClientId = self.physicsClientId)
        return pos, orn
    
    def get_link_pose(self, link_id: int):
        pos, orn = p.getLinkState(self.robot_id, link_id, physicsClientId = self.physicsClientId)[:2]
        return pos, orn
    
    def get_scene_pose(self):
        pos = self.scene_center
        orn = [0., 0., 0., 1.]
        return pos, orn

    def init_robot_pose(self, init_gripper = True):
        for config in self.robot_joint_init:
            name = config['name']
            value = config['value']
            p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict[name], targetValue = value)
        if init_gripper:
            self.init_gripper()

    def init_gripper(self):
        self.z_base = 0.07
        self.knuckle_len = 0.056
        self.x_base = 0.014
        self.angle_base = 0.63

        z_disp, x_disp = self._rev_to_pris_finger(angle=0.)

        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['finger_joint'], targetValue = 0)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['left_inner_knuckle_joint'], targetValue = 0)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['left_inner_finger_joint'], targetValue = 0)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['right_outer_knuckle_joint'], targetValue = 0)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['right_inner_knuckle_joint'], targetValue = 0)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['right_inner_finger_joint'], targetValue = 0)

        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['left_inner_finger_joint_fake'], targetValue = x_disp)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['left_finger_z_joint_fake'], targetValue = z_disp)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['right_inner_finger_joint_fake'], targetValue = -x_disp)
        p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.robot_joint_dict['right_finger_z_joint_fake'], targetValue = z_disp)


    def set_gripper_constraint(self):
        constraints = []
        # for _ in range(1):
        #     c = p.createConstraint(parentBodyUniqueId = self.robot_id,
        #                         parentLinkIndex = self.robot_joint_dict['left_outer_finger_joint'],
        #                         childBodyUniqueId = self.robot_id,
        #                         childLinkIndex = self.robot_joint_dict['left_inner_finger_joint'],
        #                         jointType=p.JOINT_POINT2POINT,
        #                         jointAxis=[0, 0, 0],
        #                         parentFramePosition=[0.001, 0, 0.0265],
        #                         childFramePosition=[0.012, 0.0, -0.013],
        #                         physicsClientId=self.physicsClientId)
        #     p.changeConstraint(c, erp=0.1, maxForce=1000, physicsClientId=self.physicsClientId)
        #     constraints.append(c)

        #     c = p.createConstraint(parentBodyUniqueId = self.robot_id,
        #                         parentLinkIndex = self.robot_joint_dict['left_inner_knuckle_joint'],
        #                         childBodyUniqueId = self.robot_id,
        #                         childLinkIndex = self.robot_joint_dict['left_inner_finger_joint'],
        #                         jointType=p.JOINT_GEAR,
        #                         jointAxis=[0, 1, 0],
        #                         parentFramePosition=[0, 0, 0],
        #                         childFramePosition=[0, 0, 0],
        #                         physicsClientId=self.physicsClientId)
        #     p.changeConstraint(c, gearRatio=1, maxForce=100, erp=2, physicsClientId=self.physicsClientId)
        #     constraints.append(c)


        #     c = p.createConstraint(parentBodyUniqueId = self.robot_id,
        #                         parentLinkIndex = self.robot_joint_dict['right_outer_finger_joint'],
        #                         childBodyUniqueId = self.robot_id,
        #                         childLinkIndex = self.robot_joint_dict['right_inner_finger_joint'],
        #                         jointType=p.JOINT_POINT2POINT,
        #                         jointAxis=[0, 0, 0],
        #                         parentFramePosition=[0.001, 0, 0.0265],
        #                         childFramePosition=[0.012, 0.0, -0.013],
        #                         physicsClientId=self.physicsClientId)
        #     p.changeConstraint(c, erp=0.1, maxForce=1000, physicsClientId=self.physicsClientId)
        #     constraints.append(c)

        #     c = p.createConstraint(parentBodyUniqueId = self.robot_id,
        #                         parentLinkIndex = self.robot_joint_dict['right_inner_knuckle_joint'],
        #                         childBodyUniqueId = self.robot_id,
        #                         childLinkIndex = self.robot_joint_dict['right_inner_finger_joint'],
        #                         jointType=p.JOINT_GEAR,
        #                         jointAxis=[0, 1, 0],
        #                         parentFramePosition=[0, 0, 0],
        #                         childFramePosition=[0, 0, 0],
        #                         physicsClientId=self.physicsClientId)
        #     p.changeConstraint(c, gearRatio=1, maxForce=100, erp=2, physicsClientId=self.physicsClientId)
        #     constraints.append(c)

        return constraints
    
    def grasp(self, val):
        self.control_target_joint_states(target_joint_names=['finger_joint'], target_pos=[val], target_vel=[], target_duration=1.0)

    def release(self):
        self.control_target_joint_states(target_joint_names=['finger_joint'], target_pos=[0.], target_vel=[], target_duration=1.0)
    
    def grasp_check(self, item_id: int) -> bool:
        if (len(p.getContactPoints(self.robot_id, item_id, self.robot_links_dict[self.lfinger_link_name], -1)) > 0) and (len(p.getContactPoints(self.robot_id, item_id, self.robot_links_dict[self.rfinger_link_name], -1)) > 0):
            return True
        else:
            return False

    def attach(self, item_id: Optional[int]) -> str:
        if item_id is None:
            assert self.target_obj_id is not None, "UR5ENV.attach(item_id): item_id must be specified."
            item_id = self.target_obj_id

        if self.grasp_check(item_id) is True:
            if not self.grasp_constraint:
                return 'ALREADY_IN_GRASP'
            else:
                grasp_constraint = []

                body_pos, body_orn = p.getBasePositionAndOrientation(bodyUniqueId=item_id, physicsClientId=self.physicsClientId)
                lfinger_pos, lfinger_orn = self.get_link_pose(link_id=self.robot_links_dict[self.lfinger_link_name])
                rel_quat = Rotation.from_quat(lfinger_orn).inv() * Rotation.from_quat(body_orn)
                rel_pos = rel_quat.apply(np.array([body_pos])-np.array([lfinger_pos]))
                rel_quat = rel_quat.as_quat()
                grasp_constraint.append(p.createConstraint(parentBodyUniqueId=self.robot_id, 
                                                           parentLinkIndex=self.robot_links_dict[self.lfinger_link_name],
                                                           childBodyUniqueId=item_id,
                                                           childLinkIndex=-1,
                                                           jointType=p.JOINT_FIXED,
                                                           jointAxis=[0., 0., 0.],
                                                           parentFramePosition=rel_pos,
                                                           childFramePosition=[0., 0., 0.],
                                                           parentFrameOrientation=rel_quat,
                                                           childFrameOrientation=[0., 0., 0., 1.],
                                                           physicsClientId=self.physicsClientId)
                )

                # body_pos, body_orn = p.getBasePositionAndOrientation(bodyUniqueId=item_id, physicsClientId=self.physicsClientId)
                # rfinger_pos, rfinger_orn = self.get_link_pose(link_id=self.robot_links_dict[self.rfinger_link_name])
                # rel_quat = Rotation.from_quat(rfinger_orn).inv() * Rotation.from_quat(body_orn)
                # rel_pos = rel_quat.apply(np.array([body_pos])-np.array([rfinger_pos]))
                # rel_quat = rel_quat.as_quat()
                # grasp_constraint.append(p.createConstraint(parentBodyUniqueId=self.robot_id, 
                #                                            parentLinkIndex=self.robot_links_dict[self.rfinger_link_name],
                #                                            childBodyUniqueId=item_id,
                #                                            childLinkIndex=-1,
                #                                            jointType=p.JOINT_FIXED,
                #                                            jointAxis=[0., 0., 0.],
                #                                            parentFramePosition=rel_pos,
                #                                            childFramePosition=[0., 0., 0.],
                #                                            parentFrameOrientation=rel_quat,
                #                                            childFrameOrientation=[0., 0., 0., 1.],
                #                                            physicsClientId=self.physicsClientId)
                # )

                self.grasp_constraint = grasp_constraint
                self.grasped_item = item_id
                return 'SUCCESS'
        else:
            return 'FAIL'

    def detach(self) -> str:
        if self.grasp_constraint:
            for constraint in self.grasp_constraint:
                p.removeConstraint(constraint, physicsClientId=self.physicsClientId)
            self.grasp_constraint = []
            self.grasped_item = None
            return 'SUCCESS'
        else:
            return 'NO_ATTACHED_OBJ'

    # def disable_gripper_self_collision(self):
    #     p.setCollisionFilterPair(self.robot_id, self.robot_id, self.robot_joint_dict['left_outer_finger_joint'], self.robot_joint_dict['left_inner_finger_joint'], 0, self.physicsClientId)
    #     p.setCollisionFilterPair(self.robot_id, self.robot_id, self.robot_joint_dict['right_outer_finger_joint'], self.robot_joint_dict['right_inner_finger_joint'], 0, self.physicsClientId)

    def gripper_mimic_constraint(self, max_torque: float = 1000):
        pos, _ = self._get_joint_state(joint_id=self.robot_joint_dict['left_inner_finger_joint'])

        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['finger_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = -pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                )
        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['left_inner_knuckle_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = -pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                )
        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['left_inner_finger_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                )

        pos, _ = self._get_joint_state(joint_id=self.robot_joint_dict['right_inner_finger_joint'])


        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['right_outer_knuckle_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = -pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                )
        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['right_inner_knuckle_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = -pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                )
        p.setJointMotorControl2(bodyIndex = self.robot_id,
                                jointIndex = self.robot_joint_dict['right_inner_finger_joint'],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = pos,
                                physicsClientId = self.physicsClientId,
                                force = max_torque
                                 )

    def control_gripper(self, target_val: float, duration: float) -> bool:
        return self.control_target_joint_states(self, target_pos = [target_val], target_vel= [0], target_duration = duration, target_joint_names = ['finger_joint'])
    








class MugEnv(UR5Env):
    def __init__(self, 
                 env_config_path: str = os.path.join(edf_env.ROOT_DIR, 'config/env_config.yaml'), 
                 scene_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/scene_camera_config.yaml'), 
                 eef_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/eef_camera_config.yaml'), 
                 monitor_cam_config_path: Optional[str] = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/monitor_camera_config.yaml'),
                 robot_path: str = os.path.join(edf_env.ROOT_DIR, 'robot/ridgeback_ur5/ridgeback_ur5_robotiq.urdf'), 
                 use_gui: bool = True, 
                 sim_freq: float = 1000,
                 debug: bool = False):
        """Pybullet environment with UR5 Robot and a mug with hanger

        Args:
            env_config_path: Path to the pybullet environment configuration file.
            scene_cam_config_path: Path to the camera configuration file for scene observation.
            eef_cam_config_path: Path to the camera configuration file for observing robot's hand (to see how the robot is gripping the object).
            robot_path: Path to the robot description file (in URDF format).
            use_gui (bool): If True, Pybullet would run with visual server. Otherewise would run with headless mode.
            sim_freq: fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).

        Attributes:
            self.physicsClientId (int): Pybullet physics server id.
            self.sim_freq (float): fps of simulation (e.g.  sim_freq=1000 means 1000 simulation steps per 1 second in simulated time).
            self.plane_id (int): Pybullet body id of the plane (ground).
            self.scene_cam_configs: A list of pybullet camera configurations for scene observation.
            self.eef_cam_configs: A list of pybullet camera configurations for observing robot's hand (to see how the robot is gripping the object).

        """

        super().__init__(env_config_path=env_config_path, scene_cam_config_path=scene_cam_config_path, eef_cam_config_path=eef_cam_config_path, monitor_cam_config_path=monitor_cam_config_path, robot_path=robot_path, use_gui=use_gui, sim_freq=sim_freq, debug=debug)
        
        if type(self) == MugEnv:
            self.reset()

    def reset(self, seed: Optional[int] = None, mug_name: str = 'train_0', hanger_name: str = 'hanger') -> bool:
        super().reset(seed=seed)

        self.mug_id = self.spawn_mug(mug_name=mug_name, 
                                     pos = self.scene_center + np.array([0, 0, 0.1]), 
                                     scale = 1.5)
        self.hanger_id = self.spawn_hanger(hanger_name=hanger_name, 
                                           pos = self.scene_center + np.array([0.27, 0, 0.]), 
                                           orn = np.array([0, 0, 1, 0]),
                                           scale = 1.0)
        self.target_obj_id = self.mug_id

        for _ in range(1000):
            self.step()

        return True

    def spawn_mug(self, mug_name: str, pos: np.ndarray, orn: Optional[np.ndarray] = None, scale: float = 1.0) -> int:
        self.mug_scate = scale
        if orn is None:
            orn = np.array([0, 0, 0, 1])
        return p.loadURDF(os.path.join(edf_env.ROOT_DIR, f"assets/mug/{mug_name}.urdf"), basePosition=pos, baseOrientation = orn, globalScaling=scale, physicsClientId = self.physicsClientId)

    def spawn_hanger(self, hanger_name: str, pos: np.ndarray, orn: Optional[np.ndarray] = None, scale: float = 1.0):
        self.hanger_scale = scale
        if orn is None:
            orn = np.array([0, 0, 0, 1])
        self.hanger_id = p.loadURDF(os.path.join(edf_env.ROOT_DIR, f"assets/hanger/{hanger_name}.urdf"), basePosition=pos, baseOrientation = orn, globalScaling=scale, physicsClientId = self.physicsClientId, useFixedBase = True)
