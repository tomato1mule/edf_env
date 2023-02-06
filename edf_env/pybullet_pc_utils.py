from typing import Union, Optional, Type, TypedDict, List, Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from edf_env.utils import CamData




##################################################################################
################################## POINTCLOUD ####################################
##################################################################################

def get_position(pix_coord_WH: np.ndarray, depth: np.ndarray, cam_view_matrix_flattened: tuple, cam_projection_matrix_flattened: tuple) -> np.ndarray:
    """Get the 3d-coordinate of specified pixels from Pybullet depth image.
       Depth values for pixels specified by pix_coord_WH are used to calculate the 3d-coordinates of those pixels.

    Returns:
        Cartesian 3d-coordinate (x,y,z) of specified pixels

        Shape::

            - Output (np.ndarray): (N_points, 3)

    Args:
        pix_coord_WH (np.ndarray): Image frame coordinates (col, row) of pixels that the points are sampled. Beware that it is not (row, col)!
        depth (np.ndarray): Raw depth image from Pybullet.
        cam_view_matrix_flattened (tuple): See the note below for detail.
        cam_projection_matrix_flattened (tuple): See the note below for detail.

        Shape::

            - pix_coord_WH: (N_points, 2)
            - depth: (H, W)
            - cam_view_matrix_flattened: Tuple[float], 16-dim (4x4 flattend)
            - cam_projection_matrix_flattened: Tuple[float], 16-dim (4x4 flattend)

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

        Depth image values are normalized to be within range [0,1].
        For the meaning of depth values, see the following link: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        Segmentation image values are non-negative integer corresponding to pybullet body ID.

    """
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
    cart_pos = pos / pos[:,3:4]                              # Homogeneous Coordinate -> Cartesian Coordinate

    return cart_pos[:,:3]                                    # Remove the last homogeneous coordinate (which is 1)

def get_inrange_indices(pos: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """ The indices of points whose poses are in the specified ranges are returned.

    Returns:
        Only the point indices inside the given range is returned.

        Shape::

            - Output (np.ndarray): (N_points_in_range,)

        Note::
            N_points_in_range <= N_points
            

    Args:
        Shape::

            - pos (np.ndarray): (N_points, 3)
            - ranges (np.ndarray): (3, 2)

    """

    xlim, ylim, zlim = ranges
    return (((pos[:,0] > xlim[0]) * (pos[:,0] < xlim[1])) * ((pos[:,1] > ylim[0]) * (pos[:,1] < ylim[1])) * ((pos[:,2] > zlim[0]) * (pos[:,2] < zlim[1]))).nonzero()[0]     # Only the points inside the given range is returned.


def rgbd_to_pc(cam_data: CamData, pix_coord_WH: np.ndarray, ranges: Optional[np.ndarray] = None, frame: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Get the point cloud from single Pybullet camera data.
       The points are ampled from specified pixels (pix_coord_WH).

    Returns:
        3d-coordinate, color and segmentation (optional).

        Shape::

            - coord (np.ndarray): (N_points, 3)
            - color (np.ndarray): (N_points, 3)
            - seg (np.ndarray[int] or None): (N_points)

    Args:
        cam_data (CamData): Pybullet camera data. See :func:`get_pybullet_cam_data` for details.
        pix_coord_WH (np.ndarray): Image frame coordinates (col, row) of pixels that the points should be sampled. Beware that it is not (row, col)!
        ranges: (np.ndarray or None): X,Y,Z box ranges of the point cloud. Any point outside this range would be discarded.
        frame ((np.ndarray, np.ndarray) or None): (pos, orn) of a reference frame. Shape: ((3,), (4,))

        Shape::

            - pix_coord_WH: (N_points, 2)
            - ranges: (3, 2)

    """

    W, H, image, depth, seg = cam_data['W'], cam_data['H'], cam_data['color'], cam_data['depth'], cam_data['seg']
    cam_view_matrix, cam_projection_matrix = cam_data['view'], cam_data['proj']
    pos = get_position(pix_coord_WH, depth, cam_view_matrix, cam_projection_matrix)
    if frame is not None:
        pos = Rotation.from_quat(frame[1]).inv().apply(pos - frame[0])

    if ranges is not None:
        inrage_indices = get_inrange_indices(pos, ranges)
        coord = pos[inrage_indices]
        inrange_pix = pix_coord_WH[inrage_indices]
    else:
        coord = pos[:]
        inrange_pix = pix_coord_WH[:]

    color = image[inrange_pix[:,1], inrange_pix[:,0]]
    if seg is not None:
        seg = seg[inrange_pix[:,1], inrange_pix[:,0]]

    return coord, color, seg

def pb_cams_to_pc(cam_data_list: List[CamData], ranges: Optional[Union[np.ndarray, list, tuple]] = None, stride: Union[np.ndarray, list, tuple] = (1,1), frame: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Get point cloud from multiple Pybullet cam data.

    Returns:
        3d-coordinate, color and segmentation (optional).

        Shape::

            - coord (np.ndarray): (N_points, 3)
            - color (np.ndarray): (N_points, 3)
            - seg (np.ndarray[int] or None): (N_points)

    Args:
        cam_data_list (List[CamData]): List of camera data. See :func:`get_pybullet_cam_data` for details as to individual camera data.
        ranges: (np.ndarray, list, tuple or None): X,Y,Z box ranges of the point cloud. Any point outside this range would be discarded.
        stride: (np.ndarray, list or tuple): Stride of sampling points from images. (1,1) means dense sampling.
        frame ((np.ndarray, np.ndarray) or None): (pos, orn) of a reference frame. Shape: ((3,), (4,))

        Shape::

            - ranges: (3, 2)
            - stride: (2,) 

    """

    ranges = np.array(ranges)

    poses, colors, segs = [], [], []
    for cam_data in cam_data_list:
        W, H = cam_data['W'], cam_data['H']
        pix_coord_WH = np.stack(np.meshgrid(np.arange(0, W, stride[1]),np.arange(0, H, stride[0]))).reshape(2,-1).T

        coord, color, seg = rgbd_to_pc(cam_data=cam_data, pix_coord_WH=pix_coord_WH, ranges=ranges, frame=frame)

        poses.append(coord)
        colors.append(color)
        if seg is not None:
            segs.append(seg)

    coord = np.concatenate(poses,axis=0)
    color = np.concatenate(colors,axis=0)
    if segs:
        seg = np.concatenate(segs,axis=0)
    else:
        seg = None

    return coord, color, seg