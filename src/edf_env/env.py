import time
import os
from typing import Union, Optional, Type, TypedDict

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from edf_env.utils import HideOutput, observe_cams, CamData, CamConfig



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
