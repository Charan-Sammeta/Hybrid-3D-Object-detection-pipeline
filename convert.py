import os
import struct
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d

# Initialize NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='D:/nuscenes', verbose=False)

# Get a random .pcd.bin file from nuScenes
pcd_bin_file = os.path.join(nusc.dataroot, nusc.get('sample_data', '9d9bf11fb0e144c8b446d54a8a00184f')['filename'])

# Load the .pcd.bin file
pc = LidarPointCloud.from_file(pcd_bin_file)
bin_pcd = pc.points.T

# Reshape and get only values for x, y, and z
bin_pcd = bin_pcd.reshape((-1, 4))[:, 0:3]

# Convert to Open3D point cloud
o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(bin_pcd)

# Save to a .pcd file with an explicit valid path
pcd_path = "D:/nuscenes/test.pcd"
success = o3d.io.write_point_cloud(pcd_path, o3d_pcd)
if not success:
    raise IOError(f"Failed to write PCD file to {pcd_path}")

# Check if file was written
if not os.path.exists(pcd_path):
    raise FileNotFoundError(f"PCD file was not created: {pcd_path}")

# Read the saved .pcd file
pcd = o3d.io.read_point_cloud(pcd_path)
out_arr = np.asarray(pcd.points)

# Debugging: Ensure the loaded PCD data is not empty
if out_arr.size == 0:
    raise ValueError("Loaded PCD file is empty! Check the file writing step.")

# Load the original point cloud data again and check if saved .pcd matches the original
pc = LidarPointCloud.from_file(pcd_bin_file)
points = pc.points.T

# Validate the point cloud data
if not np.array_equal(out_arr, points[:, :3]):
    print("Mismatch in point cloud data. Debug info:")
    print("Original points shape:", points[:, :3].shape)
    print("Loaded PCD points shape:", out_arr.shape)
    raise AssertionError("Mismatch in saved and loaded PCD data.")

print("Point cloud successfully converted and validated!")
