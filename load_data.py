from nuscenes.nuscenes import NuScenes

# Path to your dataset directory
dataset_path = "D:/NUSCENES"

# Initialize the nuScenes devkit for the mini dataset
nusc = NuScenes(version='v1.0-mini', dataroot=dataset_path, verbose=True)

# # Print scenes in the dataset
# print("Scenes in nuScenes-mini:")
# for scene in nusc.scene:
#     print(f"- Scene name: {scene['name']}, Description: {scene['description']}")


# import cv2
# import matplotlib.pyplot as plt

# # Get the first sample
# sample = nusc.sample[0]

# # Access the front camera
# camera_token = sample['data']['CAM_FRONT']
# camera_data = nusc.get('sample_data', camera_token)

# # Load and display the image
# image_path = f"{nusc.dataroot}/{camera_data['filename']}"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image)
# plt.title("Front Camera Image")
# plt.axis("off")
# plt.show()

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import matplotlib.pyplot as plt

# Get a sample from the dataset
sample = nusc.sample[0]
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)

# Load point cloud
pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

# Plot point cloud
plt.scatter(pc.points[0], pc.points[1], s=0.5)
plt.title("nuScenes LiDAR Point Cloud")
plt.show()
