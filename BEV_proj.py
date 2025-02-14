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