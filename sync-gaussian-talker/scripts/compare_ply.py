from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d

path = 'model/trial_fy/point_cloud/iteration_10000/point_cloud.ply'
path = 'model/trial_fy3/point_cloud/iteration_10000/point_cloud.ply'
path = 'model/trial_fy5/point_cloud/iteration_10000/point_cloud.ply'

def pcd_from_ply(path):
  plydata = PlyData.read(path)
  vertices = plydata['vertex']
  positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
  try:
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
  except:
    colors = np.ones_like(positions)
  # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(positions)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  return pcd

pcd = pcd_from_ply(path)

w2c = np.array([
        [ 9.9398e-01,  2.9646e-02, -1.0549e-01,  0.0000e+00],
        [ 9.2871e-03, -9.8203e-01, -1.8848e-01,  0.0000e+00],
        [-1.0918e-01,  1.8637e-01, -9.7639e-01,  0.0000e+00],
        [-8.2674e+00, -1.8702e+01,  6.0552e+02,  1.0000e+00]
        ])

full_proj = np.array([
       [ 4.2710023e+00,  1.2738580e-01, -1.0549655e-01, -1.0548600e-01],
       [ 3.9905649e-02, -4.2196712e+00, -1.8850140e-01, -1.8848255e-01],
       [-4.6912625e-01,  8.0079997e-01, -9.7649252e-01, -9.7639489e-01],
       [-3.5524197e+01, -8.0360954e+01,  6.0557007e+02,  6.0551953e+02]
       ])

# c2w = np.linalg.inv(w2c)
c2w = np.zeros_like(w2c)
R = w2c[:3, :3].T
c2w[:3, :3] = R
c2w[3, :3] = -w2c[3, :3] @ R
c2w[3, 3] = 1

homomat = np.concatenate([np.array(pcd.points), np.ones([len(pcd.points), 1])], axis=1)
new_pts = homomat @ full_proj

xys = new_pts[:, :2]
from matplotlib import pyplot as plt

ccenter = np.array([0, 0, 0, 1])
wcenter = ccenter @ c2w

plt.scatter(xys[:, 0], xys[:, 1])
plt.show()