import time
import os

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from edf_env.utils import get_image, axiscreator, img_data_to_pointcloud, HideOutput


def observe(cam_configs, physicsClientId = 0, **kwargs):
    outputs = []
    for n, config in enumerate(cam_configs):
        config = config.copy()
        if 'target_pos' in kwargs.keys():
            config['target_pos'] = kwargs['target_pos']
        data = get_image(config, physicsClientId = physicsClientId)
        W, H, image, depth = data['W'], data['H'], data['color'], data['depth']
        if 'R_sg' in kwargs.keys():
            data['R_sg'] = kwargs['R_sg']
        if 'X_sg' in kwargs.keys():
            data['X_sg'] = kwargs['X_sg']
        outputs.append(data)

    return outputs



class BulletTask():
    def __init__(self, use_gui = True, sim_freq = 1000):
        with HideOutput():
            self.physicsClientId = p.connect(p.GUI if use_gui else p.DIRECT )
        self.sim_freq = sim_freq
        self.init_task()

    def init_task(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10, physicsClientId = self.physicsClientId)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId = self.physicsClientId)
        p.setTimeStep(1/self.sim_freq)

    def reset(self, seed = None):
        raise NotImplementedError

    def close(self):
        p.disconnect()

    def observe(self):
        return observe(self.cam_configs, physicsClientId=self.physicsClientId)

    def observe_pointcloud(self):   
        data = self.observe()
        return img_data_to_pointcloud(data = data, xlim = self.xlim, ylim = self.ylim, zlim = self.zlim)
