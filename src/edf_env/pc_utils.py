from typing import Optional, Tuple

import open3d as o3d
import numpy as np


def pcd_from_numpy(coord: np.ndarray, color: Optional[np.ndarray], voxel_filter_size: Optional[float] = None):
    assert len(coord.shape) == 2, f"coord must be of shape (N_points, 3), but shape {coord.shape} is given."
    if color is None:
        raise NotImplementedError
        color = np.tile(np.array([[0.8, 0.5, 0.8]]), (coord.shape[-2],1)) # (N_points, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)

    if voxel_filter_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_filter_size)

    return pcd

def pcd_to_numpy(pcd: o3d.cuda.pybind.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors

# DEPRECATED
# def draw_pcd(input):
#     if type(input) is list:
#         assert type(input[0]) == o3d.geometry.PointCloud, f"Input must be open3d PointCloud data (or a list of open3d PointCloud data), but a list of type {type(input[0])} is given"
#         o3d.visualization.draw_geometries(input)
#     else:
#         assert type(input) == o3d.geometry.PointCloud, f"Input must be open3d PointCloud data (or a list of open3d PointCloud data), but type {type(input)} is given"
#         o3d.visualization.draw_geometries([input])

def draw_geometry(geometries):
    if not hasattr(geometries, '__iter__'):
        geometries = [geometries]
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.8, 0.8, 0.8])
    viewer.run()
    viewer.destroy_window()



def encode_pc(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    N = len(points)
    assert len(colors) == N

    colors = colors * 255

    pc = np.empty(N, dtype=[('x', np.float32),
                            ('y', np.float32),
                            ('z', np.float32),
                            ('r', np.uint8),
                            ('g', np.uint8),
                            ('b', np.uint8),
                            ])

    pc['x'] = points[:,0]
    pc['y'] = points[:,1]
    pc['z'] = points[:,2]
    pc['r'] = colors[:,0]
    pc['g'] = colors[:,1]
    pc['b'] = colors[:,2]

    return pc