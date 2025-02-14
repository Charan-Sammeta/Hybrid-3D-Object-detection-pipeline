# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud, Box

# # Configuration
# class Config:
#     # Paths
#     dataroot = 'D:\\nuscenes'  # CHANGE THIS
#     version = 'v1.0-mini'
    
#     # Data parameters
#     bev_size = (256, 256)
#     bev_resolution = 0.25  # meters per pixel
#     max_points = 30000
#     max_objects = 20
    
#     # Model parameters
#     num_classes = 10
#     point_feat_dim = 64
#     bev_feat_dim = 256
#     fusion_dim = 512
    
#     # Training parameters
#     batch_size = 4
#     lr = 1e-4
#     epochs = 50
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Utility functions
# def quaternion_yaw(q: np.ndarray) -> float:
#     """Calculate yaw angle from quaternion."""
#     return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1 - 2.0 * (q[2]**2 + q[3]**2))

# # Dataset Class
# # Update the dataset class
# class NuScenesHybridDataset(Dataset):
#     def _create_class_map(self):
#         """Create class mapping based on actual annotations."""
#         all_classes = set()
#         for sample in self.nusc.sample:
#             for ann in sample['anns']:
#                 instance = self.nusc.get('sample_annotation', ann)
#                 all_classes.add(instance['category_name'])
#         return {name: i for i, name in enumerate(sorted(all_classes))}

#     def __init__(self, nusc):
#         self.nusc = nusc
#         self.samples = nusc.sample
#         self.class_map = self._create_class_map()
#         Config.num_classes = len(self.class_map)  # Dynamic update
#         print(f"Detected {Config.num_classes} classes: {self.class_map}")
    
#     def __len__(self):
#         return len(self.samples)
    
#     def _create_bev(self, points):
#         """Create BEV representation from point cloud."""
#         bev = np.zeros((3, *Config.bev_size))
#         x, y, z, i = points
        
#         # Convert coordinates to grid indices
#         px = ((x / Config.bev_resolution) + Config.bev_size[0]//2).astype(int)
#         py = ((y / Config.bev_resolution) + Config.bev_size[1]//2).astype(int)
        
#         # Filter valid points
#         mask = (px >= 0) & (px < Config.bev_size[0]) & (py >= 0) & (py < Config.bev_size[1])
#         px, py = px[mask], py[mask]
#         z, i = z[mask], i[mask]
        
#         # Height channel
#         bev[0, py, px] = z
#         # Density channel
#         bev[1, py, px] = 1
#         # Intensity channel
#         bev[2, py, px] = i
        
#         # Normalize
#         bev[0] = (bev[0] - bev[0].min()) / (bev[0].max() - bev[0].min() + 1e-6)
#         bev[1] = np.log(bev[1] + 1)
#         bev[2] /= 255.0
        
#         return torch.tensor(bev, dtype=torch.float32)
    
#     def __getitem__(self, idx):
#         sample = self.nusc.sample[idx]
#         lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
#         # Load point cloud
#         pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_data['token']))
#         points = pc.points[:4]  # x, y, z, intensity
        
#         # Subsample points
#         if points.shape[1] > Config.max_points:
#             idxs = np.random.choice(points.shape[1], Config.max_points, replace=False)
#             points = points[:, idxs]
        
#         # Create BEV
#         bev = self._create_bev(points)
        
#         # Process annotations
#         target = {
#             'boxes': torch.zeros((Config.max_objects, 7), dtype=torch.float32),
#             'labels': torch.full((Config.max_objects,), -1, dtype=torch.long)
#         }
        
#         valid_objects = 0
#         for ann in sample['anns']:
#             if valid_objects >= Config.max_objects:
#                 break
                
#             instance = self.nusc.get('sample_annotation', ann)
#             box = self.nusc.get_box(ann)
            
#             # Convert box to BEV format
#             corners = box.corners()[:2, :4]
#             x_min, x_max = corners[0].min(), corners[0].max()
#             y_min, y_max = corners[1].min(), corners[1].max()
            
#             w = max(x_max - x_min, 0.1)
#             l = max(y_max - y_min, 0.1)
#             x = (x_min + x_max) / 2
#             y = (y_min + y_max) / 2
#             yaw = quaternion_yaw(box.orientation)
#             z = box.center[2]
#             height = box.wlh[2]
            
#             # Map class name to index
#             class_name = instance['category_name']
#             class_idx = self.class_map.get(class_name, -1)
            
#             if class_idx != -1:
#                 target['boxes'][valid_objects] = torch.tensor([x, y, w, l, yaw, z, height])
#                 target['labels'][valid_objects] = class_idx
#                 valid_objects += 1
                
#         return {
#             'bev': bev,
#             'points': torch.tensor(points.T, dtype=torch.float32),
#             'target': target
#         }

# # Model Architecture
# class PointNetBranch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(4, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, Config.point_feat_dim)
#         )
#         self.pool = nn.AdaptiveMaxPool1d(1)
        
#     def forward(self, x):
#         B, N, _ = x.shape
#         x = x.view(-1, 4)
#         x = self.mlp(x)
#         x = x.view(B, N, -1).permute(0, 2, 1)
#         return self.pool(x).squeeze(-1)

# class BEVNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, Config.bev_feat_dim, 3, padding=1),
#             nn.BatchNorm2d(Config.bev_feat_dim),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#     def forward(self, x):
#         return self.backbone(x).squeeze(-1).squeeze(-1)

# class HybridDetector(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.bev_net = BEVNet()
#         self.point_net = PointNetBranch()
        
#         self.fusion = nn.Sequential(
#             nn.Linear(Config.bev_feat_dim + Config.point_feat_dim, Config.fusion_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
        
#         # Detection heads
#         self.reg_head = nn.Sequential(
#             nn.Linear(Config.fusion_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 7 * Config.max_objects)  # Predict multiple objects
#         )
        
#         self.cls_head = nn.Sequential(
#             nn.Linear(Config.fusion_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes * Config.max_objects)
#         )
        
#     def forward(self, bev, points):
#         bev_feat = self.bev_net(bev)
#         point_feat = self.point_net(points)
#         fused = self.fusion(torch.cat([bev_feat, point_feat], dim=1))
        
#         # Reshape outputs
#         batch_size = bev.size(0)
#         reg_output = self.reg_head(fused).view(batch_size, Config.max_objects, 7)
#         cls_output = self.cls_head(fused).view(batch_size, Config.max_objects, -1)
        
#         return {
#             'cls_logits': cls_output,
#             'reg_pred': reg_output
#         }

# # Loss Function
# # In Model.py, update the HybridLoss class
# class HybridLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
#         self.reg_loss = nn.SmoothL1Loss()
        
#     def forward(self, outputs, targets):
#         # Reshape predictions
#         pred_cls = outputs['cls_logits'].view(-1, Config.num_classes)
#         pred_boxes = outputs['reg_pred'].view(-1, 7)
        
#         # Reshape targets
#         gt_boxes = targets['boxes'].view(-1, 7)
#         gt_labels = targets['labels'].view(-1)
        
#         # Filter valid targets
#         valid_mask = gt_labels != -1
#         valid_cls = gt_labels[valid_mask]
#         valid_boxes = gt_boxes[valid_mask]
        
#         # Calculate losses only if valid targets exist
#         if valid_cls.numel() > 0:
#             cls_loss = self.cls_loss(pred_cls[valid_mask], valid_cls)
#             reg_loss = self.reg_loss(pred_boxes[valid_mask], valid_boxes)
#             total_loss = cls_loss + reg_loss
#         else:
#             total_loss = torch.tensor(0.0, device=Config.device)
            
#         return total_loss
    
# # Training Pipeline
# def collate_fn(batch):
#     return {
#         'bev': torch.stack([item['bev'] for item in batch]),
#         'points': torch.nn.utils.rnn.pad_sequence(
#             [item['points'] for item in batch], batch_first=True
#         ),
#         'target': {
#             'boxes': torch.stack([item['target']['boxes'] for item in batch]),
#             'labels': torch.stack([item['target']['labels'] for item in batch])
#         }
#     }

# def train():
#     nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
#     dataset = NuScenesHybridDataset(nusc)
#     loader = DataLoader(dataset, batch_size=Config.batch_size, collate_fn=collate_fn, shuffle=True)
    
#     model = HybridDetector(num_classes=Config.num_classes).to(Config.device)
#     optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
#     criterion = HybridLoss()
    
#     # Training loop
#     for epoch in range(Config.epochs):
#         model.train()
#         total_loss = 0.0
        
#         for batch in loader:
#             # Move data to device
#             bev = batch['bev'].to(Config.device)
#             points = batch['points'].to(Config.device)
#             boxes = batch['target']['boxes'].to(Config.device)
#             labels = batch['target']['labels'].to(Config.device)
            
#             # Forward pass
#             outputs = model(bev, points)
            
#             # Calculate loss
#             loss = criterion(outputs, {'boxes': boxes, 'labels': labels})
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         print(f'Epoch {epoch+1}/{Config.epochs} | Loss: {total_loss/len(loader):.4f}')
    
#     # Save model
#     torch.save(model.state_dict(), 'hybrid_detector.pth')

# if __name__ == '__main__':
#     train()


#---------------------------------------------------------------------------------------------

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box

# Configuration
class Config:
    # Paths
    dataroot = 'D:\\nuscenes'  # CHANGE THIS to your actual Nuscenes data path
    version = 'v1.0-mini'
    
    # Data parameters
    bev_size = (256, 256)
    bev_resolution = 0.25  # meters per pixel
    max_points = 30000
    max_objects = 20
    
    # Model parameters
    num_classes = 10
    point_feat_dim = 64
    bev_feat_dim = 256
    fusion_dim = 512
    
    # Training parameters
    batch_size = 4
    lr = 1e-4
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Utility functions
def quaternion_yaw(q: np.ndarray) -> float:
    """Calculate yaw angle from quaternion."""
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1 - 2.0 * (q[2]**2 + q[3]**2))

# Dataset Class
class NuScenesHybridDataset(Dataset):
    def _create_class_map(self):
        """Create class mapping based on actual annotations."""
        all_classes = set()
        for sample in self.nusc.sample:
            for ann in sample['anns']:
                instance = self.nusc.get('sample_annotation', ann)
                all_classes.add(instance['category_name'])
        return {name: i for i, name in enumerate(sorted(all_classes))}

    def __init__(self, nusc):
        self.nusc = nusc
        self.samples = nusc.sample
        self.class_map = self._create_class_map()
        Config.num_classes = len(self.class_map)  # Dynamic update
        print(f"Detected {Config.num_classes} classes: {self.class_map}")
    
    def __len__(self):
        return len(self.samples)
    
    def _create_bev(self, points):
        """Create BEV representation from point cloud."""
        bev = np.zeros((3, *Config.bev_size))
        x, y, z, i = points
        
        # Convert coordinates to grid indices
        px = ((x / Config.bev_resolution) + Config.bev_size[0]//2).astype(int)
        py = ((y / Config.bev_resolution) + Config.bev_size[1]//2).astype(int)
        
        # Filter valid points
        mask = (px >= 0) & (px < Config.bev_size[0]) & (py >= 0) & (py < Config.bev_size[1])
        px, py = px[mask], py[mask]
        z, i = z[mask], i[mask]
        
        # Height channel
        bev[0, py, px] = z
        # Density channel
        bev[1, py, px] = 1
        # Intensity channel
        bev[2, py, px] = i
        
        # Normalize
        bev[0] = (bev[0] - bev[0].min()) / (bev[0].max() - bev[0].min() + 1e-6)
        bev[1] = np.log(bev[1] + 1)
        bev[2] /= 255.0
        
        return torch.tensor(bev, dtype=torch.float32)
    
    def __getitem__(self, idx):
        sample = self.nusc.sample[idx]
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        # Load point cloud
        pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_data['token']))
        points = pc.points[:4]  # x, y, z, intensity
        
        # Subsample points if needed
        if points.shape[1] > Config.max_points:
            idxs = np.random.choice(points.shape[1], Config.max_points, replace=False)
            points = points[:, idxs]
        
        # Create BEV image
        bev = self._create_bev(points)
        
        # Process annotations
        target = {
            'boxes': torch.zeros((Config.max_objects, 7), dtype=torch.float32),
            'labels': torch.full((Config.max_objects,), -1, dtype=torch.long)
        }
        
        valid_objects = 0
        for ann in sample['anns']:
            if valid_objects >= Config.max_objects:
                break
                
            instance = self.nusc.get('sample_annotation', ann)
            box = self.nusc.get_box(ann)
            
            # Convert box to BEV format (using its 2D footprint)
            corners = box.corners()[:2, :4]
            x_min, x_max = corners[0].min(), corners[0].max()
            y_min, y_max = corners[1].min(), corners[1].max()
            
            w = max(x_max - x_min, 0.1)
            l = max(y_max - y_min, 0.1)
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            yaw = quaternion_yaw(box.orientation)
            z = box.center[2]
            height = box.wlh[2]
            
            # Map class name to index
            class_name = instance['category_name']
            class_idx = self.class_map.get(class_name, -1)
            
            if class_idx != -1:
                target['boxes'][valid_objects] = torch.tensor([x, y, w, l, yaw, z, height])
                target['labels'][valid_objects] = class_idx
                valid_objects += 1
                
        return {
            'bev': bev,
            'points': torch.tensor(points.T, dtype=torch.float32),
            'target': target
        }

# Model Architecture
class PointNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, Config.point_feat_dim)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        B, N, _ = x.shape
        x = x.view(-1, 4)
        x = self.mlp(x)
        x = x.view(B, N, -1).permute(0, 2, 1)
        return self.pool(x).squeeze(-1)

class BEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, Config.bev_feat_dim, 3, padding=1),
            nn.BatchNorm2d(Config.bev_feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1).squeeze(-1)

class HybridDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bev_net = BEVNet()
        self.point_net = PointNetBranch()
        
        self.fusion = nn.Sequential(
            nn.Linear(Config.bev_feat_dim + Config.point_feat_dim, Config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Detection heads
        self.reg_head = nn.Sequential(
            nn.Linear(Config.fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 7 * Config.max_objects)  # Predict multiple objects per sample
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(Config.fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes * Config.max_objects)
        )
        
    def forward(self, bev, points):
        bev_feat = self.bev_net(bev)
        point_feat = self.point_net(points)
        fused = self.fusion(torch.cat([bev_feat, point_feat], dim=1))
        
        # Reshape outputs for multiple objects
        batch_size = bev.size(0)
        reg_output = self.reg_head(fused).view(batch_size, Config.max_objects, 7)
        cls_output = self.cls_head(fused).view(batch_size, Config.max_objects, -1)
        
        return {
            'cls_logits': cls_output,
            'reg_pred': reg_output
        }

# Loss Function
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.reg_loss = nn.SmoothL1Loss()
        
    def forward(self, outputs, targets):
        # Reshape predictions
        pred_cls = outputs['cls_logits'].view(-1, Config.num_classes)
        pred_boxes = outputs['reg_pred'].view(-1, 7)
        
        # Reshape targets
        gt_boxes = targets['boxes'].view(-1, 7)
        gt_labels = targets['labels'].view(-1)
        
        # Filter valid targets
        valid_mask = gt_labels != -1
        valid_cls = gt_labels[valid_mask]
        valid_boxes = gt_boxes[valid_mask]
        
        # Calculate losses only if valid targets exist
        if valid_cls.numel() > 0:
            cls_loss = self.cls_loss(pred_cls[valid_mask], valid_cls)
            reg_loss = self.reg_loss(pred_boxes[valid_mask], valid_boxes)
            total_loss = cls_loss + reg_loss
        else:
            total_loss = torch.tensor(0.0, device=Config.device)
            
        return total_loss

# Training Pipeline
def collate_fn(batch):
    return {
        'bev': torch.stack([item['bev'] for item in batch]),
        'points': torch.nn.utils.rnn.pad_sequence(
            [item['points'] for item in batch], batch_first=True
        ),
        'target': {
            'boxes': torch.stack([item['target']['boxes'] for item in batch]),
            'labels': torch.stack([item['target']['labels'] for item in batch])
        }
    }

def train():
    nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
    dataset = NuScenesHybridDataset(nusc)
    loader = DataLoader(dataset, batch_size=Config.batch_size, collate_fn=collate_fn, shuffle=True)
    
    model = HybridDetector(num_classes=Config.num_classes).to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    criterion = HybridLoss()
    
    # Training loop
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in loader:
            # Move data to device
            bev = batch['bev'].to(Config.device)
            points = batch['points'].to(Config.device)
            boxes = batch['target']['boxes'].to(Config.device)
            labels = batch['target']['labels'].to(Config.device)
            
            # Forward pass
            outputs = model(bev, points)
            
            # Calculate loss
            loss = criterion(outputs, {'boxes': boxes, 'labels': labels})
            
            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{Config.epochs} | Loss: {total_loss/len(loader):.4f}')
    
    # Save model weights
    torch.save(model.state_dict(), 'hybrid_detector.pth')

if __name__ == '__main__':
    train()
