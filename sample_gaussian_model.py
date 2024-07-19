import torch
import numpy as np
from plyfile import PlyData, PlyElement

from utils.general_utils import build_rotation
from utils.system_utils import mkdir_p
import sys
import os


if len(sys.argv) < 2:
    print("Usage: python sample_gaussian_model.py [ gaussian model path ] \nExample: python sample_gaussian_model.py Z:\\2b46a838-e\\point_cloud\\iteration_30000\\")
    exit(0)
raw_path = sys.argv[1]
# raw_path = r"${your model path}\point_cloud\iteration_30000\"

if not os.path.exists(os.path.join(raw_path, 'point_cloud.ply')):
    print(f"The point_cloud.ply file does not exist in the directory [\"{raw_path}\"].")
    exit(0)

plydata = PlyData.read(os.path.join(raw_path, 'point_cloud.ply'))
print("Read point_cloud.ply succeed.")
xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
scales = np.zeros((xyz.shape[0], len(scale_names)))
for idx, attr_name in enumerate(scale_names):
         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
rots = np.zeros((xyz.shape[0], len(rot_names)))
for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
print("Read gaussian model succeed.")

_xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
_opacity = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device="cuda"))
_scaling = torch.exp(torch.tensor(scales, dtype=torch.float, device="cuda"))
_rotation = torch.nn.functional.normalize(torch.tensor(rots, dtype=torch.float, device="cuda"))


N=3
print(f"Sampling Gaussian points randomly. N={N}")
stds = _scaling.repeat(N, 1)
means = torch.zeros((stds.size(0), 3), device="cuda")
samples = torch.normal(mean=means, std=stds)
rots = build_rotation(_rotation).repeat(N, 1, 1)
new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _xyz.repeat(N, 1)


mkdir_p(os.path.dirname(os.path.join(raw_path, 'point_cloud_sample.ply')))

xyz = new_xyz.cpu().numpy()
normals = np.zeros_like(xyz)

list_of_attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']

dtype_full = [(attribute, 'f4') for attribute in list_of_attributes]

elements = np.empty(xyz.shape[0], dtype=dtype_full)
# attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, score), axis=1)
attributes = np.concatenate((xyz, normals), axis=1)
elements[:] = list(map(tuple, attributes))
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(os.path.join(raw_path, 'point_cloud_sample.ply'))
print(f"Sample over. The point cloud file has been saved to \"{os.path.join(raw_path, 'point_cloud_sample.ply')}\"")