import os
from nuscenes.nuscenes import NuScenes

# Path to your dataset
DATAROOT = 'D:\\nuscenes'

# Initialize NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT)

# List available scenes
print("Available scenes:")
for scene in nusc.scene:
    print(f"Scene name: {scene['name']}, Token: {scene['token']}")

# Specify the scenes for mini_val
mini_val_scene_names = ['scene-0061', 'scene-0103', 'scene-0553']  # Choose scenes from the available ones
mini_val_tokens = set()

# Extract tokens for the specified scenes
for scene in nusc.scene:
    if scene['name'] in mini_val_scene_names:
        print(f"Processing scene: {scene['name']}")
        sample_token = scene['first_sample_token']
        while sample_token:
            mini_val_tokens.add(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']

# Check if tokens were added
if not mini_val_tokens:
    print("No tokens found for the specified scenes. Check the scene names.")
else:
    print(f"Found {len(mini_val_tokens)} tokens for mini_val.")

# Create the split directory
split_dir = os.path.join(DATAROOT, 'v1.0-mini/split')
os.makedirs(split_dir, exist_ok=True)

# Save tokens to mini_val.txt
output_path = os.path.join(split_dir, 'mini_val.txt')
with open(output_path, 'w') as f:
    for token in mini_val_tokens:
        f.write(f"{token}\n")

print(f"mini_val.txt saved to {output_path}")
