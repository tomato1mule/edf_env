import sys, os
from typing import Union, Optional, Type, TypedDict

import pybullet as p
import numpy as np



##################################################################################
################################## Observation ###################################
##################################################################################

class CamConfig(TypedDict):
    distance: float
    ypr: list[float]
    W: int
    H: int
    up: list[float]
    up_axis_idx: int
    near: float
    far : float
    fov : float
    target_pos: Optional[np.ndarray]

class CamData(TypedDict):
    color: np.ndarray
    depth: Optional[np.ndarray]
    seg: Optional[np.ndarray]
    view: Optional[np.ndarray]
    proj: Optional[np.ndarray]
    W: int
    H: int

def get_pybullet_cam_data(cam_config: CamConfig, physicsClientId: int = 0) -> CamData:
    """Get the camera data of a speficied camera configuration from Pybullet server

    Returns:
        the camera data for cam_config, which includes color/depth/segmentation images and camera parameters.

        Shape::

            - Output['color']: (H, W, 3)
            - Output['depth']: (H, W)
            - Output['seg']: (H, W)
            - Output['view']: Tuple[float], 16-dim (4x4 flattend)
            - Output['proj']: Tuple[float], 16-dim (4x4 flattend)
            - Output['H']: int
            - Output['W']: int


    Args:
        cam_config (CamConfig): A typed dict that specifies camera configurations.
        physicsClientId (int): Pybullet server ID.

    Note:
        Camera frame
        
            y
            |
            |
            o-----x
        (z direction coming out of the screen -> right hand rule,  thus z always be negative)

        According to https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        First, pybullet uses the notation of OpenGL so it is using a major-column order (read more). 
        Meaning the first element while indexing is the column and not the row. 
        Therefore the actual projection matrix from pybullet should be "TRANSPOSED".

        Color and depth image values are normalized to be within range [0,1].
        For the meaning of depth values, see the following link: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        Segmentation image values are non-negative integer corresponding to pybullet body ID.

    """
    cam_target_pos = cam_config['target_pos']
    cam_distance = cam_config['distance']
    cam_yaw, cam_pitch, cam_roll = cam_config['ypr']
    cam_width, cam_height = cam_config['W'], cam_config['H']

    cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = cam_config['up'], cam_config['up_axis_idx'], cam_config['near'], cam_config['far'], cam_config['fov']

    cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
    cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width*1./cam_height, cam_near_plane, cam_far_plane)

    W, H, image, depth, seg = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix, physicsClientId = physicsClientId)
    image = image[:,:,:3] / 255


    cam_data: CamData = {'color': image, 'depth': depth, 'seg': seg, 'view': cam_view_matrix, 'proj': cam_projection_matrix, 'W': W, 'H': H}
    return cam_data


def observe_cams(cam_configs: list[CamConfig], target_pos: Optional[np.ndarray] = None, physicsClientId: int = 0) -> list[CamData]:
    """Observes multiple pybullet virtual camera data from list of camera configurations.
    If target_pos is specified, all the cameras will look at the same fixation point in the specificed target position.

    Returns:
        A list of camera data for cam_configs. See :func:`get_pybullet_cam_data` for details as to individual camera config/data.


    Args:
        cam_config (List[CamConfig]): A typed dict that specifies camera configurations.
        target_pos (Optional[np.ndarray]): If target_pos is specified, all the cameras will look at the same fixation point in the specificed target position.
        physicsClientId (int): Pybullet server ID.

        Shape::

            - target_pos: (3,)
            
    """

    outputs: list[CamData] = []
    for n, config in enumerate(cam_configs):
        config = config.copy()
        if target_pos is not None:
            config['target_pos'] = target_pos.copy()
        else:
            assert config['target_pos'] is not None, "Target position of the camera must be specified."
        data: CamData = get_pybullet_cam_data(config, physicsClientId = physicsClientId)
        outputs.append(data)

    return outputs








##################################################################################
################################## POINTCLOUD ####################################
##################################################################################

# camera frame
#
#     y
#     |
#     |
#     o-----x
# (z direction out of paper -> right hand rule,  thus z always be negative)


# According to https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
# First, pybullet uses the notation of OpenGL so it is using a major-column order (read more). 
# Meaning the first element while indexing is the column and not the row. 
# Therefore the actual projection matrix from pybullet should be "TRANSPOSED".


# Code borrowed from benbo yang:   https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
def get_position(pix_coord_WH, depth, cam_view_matrix_flattened, cam_projection_matrix_flattened):
    assert len(pix_coord_WH.shape) == 2 and pix_coord_WH.shape[1] == 2
    assert np.max(pix_coord_WH[:,0]) < depth.shape[1] and np.max(pix_coord_WH[:,1]) < depth.shape[0]

    projectionMatrix = np.asarray(cam_projection_matrix_flattened).reshape([4,4],order='F')     # .reshape([4,4],order='F')  is equivalent to .reshape([4,4]).T    (openGL formalizm uses faster index for row, not column)
    viewMatrix = np.asarray(cam_view_matrix_flattened).reshape([4,4],order='F')                 # .reshape([4,4],order='F')  is equivalent to .reshape([4,4]).T    (openGL formalizm uses faster index for row, not column)
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))


    H,W = depth.shape 
    w,h = pix_coord_WH[:,0], pix_coord_WH[:,1]

    x = (2 * w - W) / W
    y = (2 * h - H) / H * -1
    z = 2*depth[h, w] - 1

    pixPos = np.stack((x,y,z,np.ones(x.shape[0])), axis=-1)
    pos = (tran_pix_world @ pixPos.T).T
    cart_pos = pos / pos[:,3:4]
    
    return cart_pos[:,:3]

def rgbd_to_pointcloud(data, pix_coord_WH, xlim = (-0.5,0.5), ylim = (-0.5,0.5), zlim = (-0.2,0.8)):
    W, H, image, depth = data['W'], data['H'], data['color'], data['depth']
    cam_view_matrix, cam_projection_matrix = data['view'], data['proj']
    pos = get_position(pix_coord_WH, depth, cam_view_matrix, cam_projection_matrix)

    in_playground_idx = (((pos[:,0] > xlim[0]) * (pos[:,0] < xlim[1])) * ((pos[:,1] > ylim[0]) * (pos[:,1] < ylim[1])) * ((pos[:,2] > zlim[0]) * (pos[:,2] < zlim[1]))).nonzero()[0]
    coord = pos[in_playground_idx]
    pix_in_playground = pix_coord_WH[in_playground_idx]
    color = image[pix_in_playground[:,1], pix_in_playground[:,0]]

    return coord, color

def img_data_to_pointcloud(data, xlim, ylim, zlim, stride=(1,1)):
    poses, colors= [], []
    for datum in data:
        W, H = datum['W'], datum['H']
        if 'R_sg' in datum.keys():
            assert 'X_sg' in datum.keys()
            R_sg = datum['R_sg']
            X_sg = datum['X_sg']
        else:
            X_sg = np.zeros((3,2))

        pix_coord_WH = np.stack(np.meshgrid(np.arange(0, W, stride[1]),np.arange(0, H, stride[0]))).reshape(2,-1).T

        coord, color = rgbd_to_pointcloud(datum, pix_coord_WH, xlim + X_sg[0], ylim + X_sg[1], zlim + X_sg[2])

        poses.append(coord)
        colors.append(color)

    coord = np.concatenate(poses,axis=0)
    color = np.concatenate(colors,axis=0)

    if 'R_sg' in datum.keys():
        coord = (coord - X_sg) @ R_sg

    outputs =  {'coord': coord,
                'color': color,
                'ranges': np.array([xlim.copy(), ylim.copy(), zlim.copy()])}

    if 'R_sg' in datum.keys():
        outputs['R_sg'] = R_sg
        outputs['X_sg'] = X_sg

    return outputs




##################################################################################
################################# VISUALIZE ######################################
##################################################################################


# code borrowed from cvshah:   https://github.com/bulletphysics/bullet3/discussions/3867

def axiscreator(bodyId: int, linkId: int = -1, offset: Union[list[float], np.ndarray] = [0, 0, 0], physicsClientId: int = 0) -> list[int]:
    print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB || offset: {offset}')
    offset = np.array(offset)

    x_axis_id: int = p.addUserDebugLine(lineFromXYZ          = offset                         ,
                                        lineToXYZ            = offset + np.array([0.2, 0, 0]) ,
                                        lineColorRGB         = [1, 0, 0]  ,
                                        lineWidth            = 5          ,
                                        lifeTime             = 0          ,
                                        parentObjectUniqueId = bodyId     ,
                                        parentLinkIndex      = linkId     ,
                                        physicsClientId      = physicsClientId)

    y_axis_id: int = p.addUserDebugLine(lineFromXYZ          = offset                         ,
                                        lineToXYZ            = offset + np.array([0, 0.2, 0]) ,
                                        lineColorRGB         = [0, 1, 0]  ,
                                        lineWidth            = 5          ,
                                        lifeTime             = 0          ,
                                        parentObjectUniqueId = bodyId     ,
                                        parentLinkIndex      = linkId     ,
                                        physicsClientId      = physicsClientId)

    z_axis_id: int = p.addUserDebugLine(lineFromXYZ          = offset                         ,
                                        lineToXYZ            = offset + np.array([0, 0, 0.2]) ,
                                        lineColorRGB         = [0, 0, 1]  ,
                                        lineWidth            = 5          ,
                                        lifeTime             = 0          ,
                                        parentObjectUniqueId = bodyId     ,
                                        parentLinkIndex      = linkId     ,
                                        physicsClientId      = physicsClientId)
    return [x_axis_id, y_axis_id, z_axis_id]


##################################################################################
###################################### MISC ######################################
##################################################################################


# Borrowed from pybullet-planning
class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    '''
    A context manager that block stdout for its scope, usage:
    with HideOutput():
        os.system('ls -l')
    '''
    DEFAULT_ENABLE = True
    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        #self.fd = sys.stdout.fileno()
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno) # Added