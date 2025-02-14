import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Initialize dataset
nusc = NuScenes(version='v1.0-mini', dataroot='D:\\nuscenes', verbose=True)

# Choose a sample
sample_token = nusc.sample[0]['token']
sample = nusc.get('sample', sample_token)
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

# Load point cloud
pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_data['token']))

def create_bev_nuscenes(points, grid_size=100, resolution=0.1, z_offset=2.0):
    """
    Create BEV map for NuScenes LiDAR data
    :param points: (4, N) array of (x, y, z, intensity)
    :param grid_size: Size of grid in meters (x and y)
    :param resolution: Meters per pixel
    :param z_offset: Height offset for visualization
    """
    # NuScenes coordinate system: x=forward, y=left, z=up
    x = points[0]
    y = points[1]
    z = points[2]
    intensity = points[3]

    # Filter points (keep points within 50m)
    mask = (np.abs(x) < grid_size/2) & (np.abs(y) < grid_size/2) & (np.sqrt(x**2 + y**2) < 50)
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    # Create grid
    grid_pixels = int(grid_size / resolution)
    bev = np.zeros((grid_pixels, grid_pixels, 3), dtype=np.float32)

    # Convert coordinates to grid indices
    col = ((x + grid_size/2) / resolution).astype(int)
    row = ((y + grid_size/2) / resolution).astype(int)

    # Filter valid indices
    valid = (col >= 0) & (col < grid_pixels) & (row >= 0) & (row < grid_pixels)
    col, row = col[valid], row[valid]
    z, intensity = z[valid], intensity[valid]

    # Height channel (normalized)
    bev[row, col, 0] = np.maximum(bev[row, col, 0], z + z_offset)
    
    # Density channel
    bev[row, col, 1] += 1  # Count points
    
    # Intensity channel
    bev[row, col, 2] = intensity

    # Post-processing
    bev[..., 0] = (bev[..., 0] - bev[..., 0].min()) / (bev[..., 0].max() - bev[..., 0].min())  # Height
    bev[..., 1] = np.log(bev[..., 1] + 1) / np.log(256)  # Density
    bev[..., 2] = bev[..., 2] / 255  # Intensity

    return bev

# Generate BEV
bev = create_bev_nuscenes(pc.points)

# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(bev, aspect='equal')
plt.title('NuScenes BEV Projection')
plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def create_range_image_nuscenes(points, num_beams=32, fov=(-30, 10), height=64, width=1024):
    """
    Create range image for NuScenes LiDAR (32 beams)
    :param points: (4, N) array of (x, y, z, intensity)
    :param num_beams: 32 for NuScenes LIDAR_TOP
    :param fov: Vertical field of view (degrees)
    """
    x, y, z = points[:3]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Convert to spherical coordinates
    theta = np.arctan2(y, x)  # Azimuth [-pi, pi]
    phi = np.arcsin(z / r)    # Elevation [-pi/2, pi/2]

    # Convert to degrees
    theta_deg = np.degrees(theta) % 360
    phi_deg = np.degrees(phi)

    # Initialize range image
    range_image = np.zeros((height, width), dtype=np.float32)
    intensity_image = np.zeros((height, width), dtype=np.float32)

    # Discretize angles
    col = np.floor((theta_deg / 360) * width).astype(int)
    row = np.floor((phi_deg - fov[0]) / (fov[1] - fov[0]) * height).astype(int)

    # Filter valid indices
    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    row, col = row[valid], col[valid]
    r_valid = r[valid]
    intensity_valid = points[3][valid]

    # Keep closest points (handle occlusion)
    for r_val, c, i in zip(r_valid, col, intensity_valid):
        if range_image[row[c], c] == 0 or r_val < range_image[row[c], c]:
            range_image[row[c], c] = r_val
            intensity_image[row[c], c] = i

    # Normalize
    range_image = range_image / 100  # Assuming max range 100m
    intensity_image = intensity_image / 255

    return np.stack([range_image, intensity_image], axis=-1)

# Generate range image
range_img = create_range_image_nuscenes(pc.points)

# Visualize
plt.figure(figsize=(15, 4))
plt.subplot(121)
plt.title('Range Channel')
plt.imshow(range_img[..., 0], cmap='viridis')
plt.subplot(122)
plt.title('Intensity Channel')
plt.imshow(range_img[..., 1], cmap='gray')
plt.show()