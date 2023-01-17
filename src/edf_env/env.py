import time
import os, sys
from typing import Union, Optional, Type, TypedDict, Any

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import yaml

import edf_env
from edf_env.utils import HideOutput, observe_cams, CamData, CamConfig, load_yaml
from edf_env.pybullet_pc_utils import pb_cams_to_pc





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

    def load_cam_config(self, cam_config_path: str) -> list[CamConfig]:
        """Loads list of pybullet camera configs from yaml file path."""
        cam_configs: list[CamConfig] = load_yaml(cam_config_path)
        return cam_configs

    def observe_cams(self, cam_configs: list[CamConfig], target_pos: Optional[np.ndarray] = None) -> list[CamData]:
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
        
        return observe_cams(cam_configs=cam_configs, target_pos=target_pos, physicsClientId=self.physicsClientId)
        
    def observe(self):
        raise NotImplementedError



class UR5Env(BulletEnv):
    def __init__(self, 
                 env_config_path: Optional[str] = None, 
                 scene_cam_config_path: Optional[str] = None, 
                 grasp_cam_config_path: Optional[str] = None, 
                 robot_path: Optional[str] = None, 
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

        if env_config_path is None:
            env_config_path = os.path.join(edf_env.ROOT_DIR, 'config/env_config.yaml')
        self.load_env_config(config_path=env_config_path)

        ### Temporary ###
        if scene_cam_config_path is None:
            scene_cam_config_path = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/scene_camera_config.yaml')
        self.scene_cam_configs = self.load_cam_config(cam_config_path=scene_cam_config_path)

        if grasp_cam_config_path is None:
            grasp_cam_config_path = os.path.join(edf_env.ROOT_DIR, 'config/camera_config/grasp_camera_config.yaml')
        self.grasp_cam_configs = self.load_cam_config(cam_config_path=grasp_cam_config_path)
        #################

        if robot_path is None:
            robot_path = os.path.join(edf_env.ROOT_DIR, 'robot/ridgeback_ur5/ridgeback_ur5.urdf')
        self.load_robot(urdf_path=robot_path)
        
    def load_env_config(self, config_path: str):
        """Loads environment config from yaml file path."""
        config: dict[str, Any] = load_yaml(config_path)
            
        scene_config: dict[str, Any] = config['scene_config']
        self.scene_ranges: np.ndarray = np.array(scene_config['scene_ranges']) # [[x_min, x_max], [y_min, y_max], [z_min, z_max]];  Shape: (3,2)
        self.scene_center: np.ndarray = np.array(scene_config['scene_center']) # [xc,yc,zc]: Shape: (3,)
        if scene_config['relative_range'] is True:
            self.scene_ranges = self.scene_ranges + np.stack([self.scene_center, self.scene_center], axis=1)
        else:
            raise NotImplementedError

        robot_config: dict[str, Any] = config['robot_config']
        self.robot_base_pose_init: dict[str, Optional[np.ndarray]] = robot_config['robot_base_pose_init']
        if self.robot_base_pose_init['pos'] is None:
            self.robot_base_pose_init['pos'] = np.array([0.0, 0.0, 0.0])
        if self.robot_base_pose_init['orn'] is None:
            self.robot_base_pose_init['orn'] = np.array([0.0, 0.0, 0.0, 1.0])



    def load_robot(self, urdf_path: str):
        """Loads list of pybullet camera configs from yaml file path."""
        p.loadURDF(fileName=urdf_path, physicsClientId=self.physicsClientId, basePosition = self.robot_base_pose_init['pos'], baseOrientation = self.robot_base_pose_init['orn'])



        ### Temporary ###
    def observe_scene(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], list[CamData]]:
        cam_data_list: list[CamData] = self.observe_cams(cam_configs=self.scene_cam_configs, target_pos=self.scene_center)
        pc_coord, pc_color, pc_seg = pb_cams_to_pc(cam_data_list=cam_data_list, ranges=self.scene_ranges)

        return pc_coord, pc_color, pc_seg, cam_data_list
        #################