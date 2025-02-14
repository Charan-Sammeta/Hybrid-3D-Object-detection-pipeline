# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from Model import (
#     Config,
#     NuScenesHybridDataset,
#     HybridDetector,
#     quaternion_yaw,
#     collate_fn
# )
# from nuscenes import NuScenes
# from nuscenes.eval.detection.evaluate import NuScenesEval
# from nuscenes.eval.detection.config import config_factory
# from tqdm import tqdm
# from torchvision.ops import nms

# class NuScenesEvaluator:
#     def __init__(self, model_path='hybrid_detector.pth'):
#         self.device = Config.device
#         self.nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
#         self.dataset = NuScenesHybridDataset(self.nusc)
        
#         # Load model
#         self.model = HybridDetector(num_classes=Config.num_classes).to(self.device)
#         self.model.load_state_dict(torch.load(model_path))
#         self.model.eval()
        
#     def predict(self, sample_idx, conf_threshold=0.1, nms_threshold=0.4):
#         """Run inference on a single sample."""
#         sample = self.dataset[sample_idx]
#         bev = sample['bev'].unsqueeze(0).to(self.device)
#         points = sample['points'].unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             outputs = self.model(bev, points)
        
#         return self._decode_outputs(outputs, sample, conf_threshold, nms_threshold)
    
#     def _decode_outputs(self, outputs, sample, conf_threshold, nms_threshold):
#         """Convert model outputs to usable bounding boxes."""
#         # Get predictions
#         pred_boxes = outputs['reg_pred'][0].cpu().numpy()  # Shape: [num_predictions, 7]
#         pred_cls = torch.softmax(outputs['cls_logits'], dim=1)[0].cpu().numpy()  # Shape: [num_classes]
        
#         # Handle single prediction vs multiple predictions
#         if pred_boxes.ndim == 1:
#             pred_boxes = pred_boxes.reshape(1, -1)
#             pred_cls = pred_cls.reshape(1, -1)
        
#         # Filter by confidence
#         conf_scores = np.max(pred_cls, axis=1)
#         valid_mask = conf_scores > conf_threshold
        
#         pred_boxes = pred_boxes[valid_mask]
#         pred_cls = pred_cls[valid_mask]
        
#         # Rest of the decoding remains the same
#         decoded_boxes = []
#         for box, cls_prob in zip(pred_boxes, pred_cls):
#             x, y, w, l, yaw, z, height = box
#             class_id = np.argmax(cls_prob)
            
#             decoded_boxes.append({
#                 'x': (x * Config.bev_resolution) - (Config.bev_size[0]//2 * Config.bev_resolution),
#                 'y': (y * Config.bev_resolution) - (Config.bev_size[1]//2 * Config.bev_resolution),
#                 'w': w,
#                 'l': l,
#                 'yaw': yaw,
#                 'z': z,
#                 'height': height,
#                 'class_id': class_id,
#                 'confidence': np.max(cls_prob)
#             })
        
#         # Apply NMS
#         if decoded_boxes:
#             boxes_array = np.array([[b['x'], b['y'], b['w'], b['l']] for b in decoded_boxes])
#             scores = np.array([b['confidence'] for b in decoded_boxes])
#             keep = nms(
#                 torch.tensor(boxes_array),
#                 torch.tensor(scores),
#                 nms_threshold
#             )
#             decoded_boxes = [decoded_boxes[i] for i in keep.numpy()]
        
#         return decoded_boxes, sample
    
#     def visualize(self, pred_boxes, sample):
#         """Visualize predictions vs ground truth."""
#         bev = sample['bev'].numpy().transpose(1, 2, 0)
        
#         plt.figure(figsize=(20, 10))
        
#         # Plot predictions
#         plt.subplot(121)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Predictions')
#         self._draw_boxes(pred_boxes, color='red')
        
#         # Plot ground truth
#         plt.subplot(122)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Ground Truth')
#         self._draw_boxes(self._get_gt_boxes(sample), color='green')
        
#         plt.show()
    
#     def _draw_boxes(self, boxes, color):
#         """Helper function to draw boxes on BEV."""
#         for box in boxes:
#             x = (box['x'] + Config.bev_size[0]//2 * Config.bev_resolution) / Config.bev_resolution
#             y = (box['y'] + Config.bev_size[1]//2 * Config.bev_resolution) / Config.bev_resolution
#             w = box['w'] / Config.bev_resolution
#             l = box['l'] / Config.bev_resolution
            
#             rect = plt.Rectangle(
#                 (x - w/2, y - l/2), w, l,
#                 angle=np.degrees(box['yaw']),
#                 rotation_point='center',
#                 fill=False,
#                 color=color,
#                 linewidth=2
#             )
#             plt.gca().add_patch(rect)
#             if 'confidence' in box:
#                 plt.text(x, y, 
#                          f"{list(self.dataset.class_map.keys())[box['class_id']]}\n{box['confidence']:.2f}",
#                          color='white', fontsize=8, ha='center')
    
#     def _get_gt_boxes(self, sample):
#         """Convert ground truth boxes to visualization format."""
#         gt_boxes = []
#         for box, label in zip(sample['target']['boxes'], sample['target']['labels']):
#             if label == -1:
#                 continue
#             gt_boxes.append({
#                 'x': box[0].item(),
#                 'y': box[1].item(),
#                 'w': box[2].item(),
#                 'l': box[3].item(),
#                 'yaw': box[4].item(),
#                 'z': box[5].item(),
#                 'height': box[6].item(),
#                 'class_id': label.item()
#             })
#         return gt_boxes
    
#     def evaluate(self, output_file='results.json'):
#         """Full evaluation using NuScenes metrics."""
#         results = []
        
#         for sample_idx in tqdm(range(len(self.dataset))):
#             sample = self.dataset[sample_idx]
#             pred_boxes, _ = self.predict(sample_idx)
            
#             # Convert to NuScenes format
#             sample_token = self.nusc.sample[sample_idx]['token']
#             for box in pred_boxes:
#                 results.append({
#                     'sample_token': sample_token,
#                     'translation': [box['x'], box['y'], box['z']],
#                     'size': [box['l'], box['w'], box['height']],
#                     'rotation': [np.cos(box['yaw']/2), 0, 0, np.sin(box['yaw']/2)],
#                     'detection_name': list(self.dataset.class_map.keys())[box['class_id']],
#                     'detection_score': box['confidence']
#                 })
        
#         # Save results
#         with open(output_file, 'w') as f:
#             json.dump(results, f)
            
#         # Run official evaluation
#         nusc_eval = NuScenesEval(
#             self.nusc,
#             config=config_factory('detection_cvpr_2019'),
#             result_path=output_file,
#             eval_set='mini_val',
#             output_dir='./eval_results'
#         )
#         return nusc_eval.main()

# if __name__ == '__main__':
#     evaluator = NuScenesEvaluator()
    
#     # Example: Visualize sample 0
#     pred_boxes, sample = evaluator.predict(0)
#     evaluator.visualize(pred_boxes, sample)
    
#     # Full evaluation
#     metrics = evaluator.evaluate()
#     print("Evaluation Metrics:")
#     print(metrics)

#-----------------------------------------------------------------------
#----------------------------------------------------------------------=

# import os
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from Model import (
#     Config,
#     NuScenesHybridDataset,
#     HybridDetector,
#     quaternion_yaw,
#     collate_fn
# )
# from nuscenes.nuscenes import NuScenes
# from nuscenes.eval.detection.evaluate import NuScenesEval
# from nuscenes.eval.detection.config import config_factory
# from tqdm import tqdm
# from torchvision.ops import nms

# # --- Set evaluation split ---
# Config.eval_split = 'mini_val'

# def load_split_tokens(split_file):
#     """Load sample tokens from a split file (one token per line)."""
#     with open(split_file, 'r') as f:
#         tokens = [line.strip() for line in f if line.strip()]
#     return set(tokens)

# def get_scene_sample_tokens(nusc, scene_token):
#     """Traverse a scene’s sample chain and return all sample tokens for that scene."""
#     tokens = []
#     scene = nusc.get('scene', scene_token)
#     sample_token = scene['first_sample_token']
#     while sample_token:
#         tokens.append(sample_token)
#         sample = nusc.get('sample', sample_token)
#         sample_token = sample['next']
#     return tokens

# class NuScenesEvaluator:
#     def __init__(self, model_path='hybrid_detector.pth'):
#         self.device = Config.device
#         self.nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
#         self.dataset = NuScenesHybridDataset(self.nusc)
        
#         # Load the trained model
#         self.model = HybridDetector(num_classes=Config.num_classes).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
        
#         # Mapping from your dataset's raw class names to the 10 names expected by NuScenes evaluation.
#         self.detection_name_mapping = {
#             'human.pedestrian.adult': 'pedestrian',
#             'human.pedestrian.child': 'pedestrian',
#             'human.pedestrian.construction_worker': 'pedestrian',
#             'human.pedestrian.personal_mobility': 'pedestrian',
#             'human.pedestrian.police_officer': 'pedestrian',
#             'movable_object.barrier': 'barrier',
#             'movable_object.debris': 'barrier',
#             'movable_object.pushable_pullable': 'barrier',
#             'movable_object.trafficcone': 'traffic_cone',
#             'static_object.bicycle_rack': 'bicycle',
#             'vehicle.bicycle': 'bicycle',
#             'vehicle.bus.bendy': 'bus',
#             'vehicle.bus.rigid': 'bus',
#             'vehicle.car': 'car',
#             'vehicle.construction': 'construction_vehicle',
#             'vehicle.motorcycle': 'motorcycle',
#             'vehicle.trailer': 'trailer',
#             'vehicle.truck': 'truck'
#         }
        
#     def predict(self, sample_idx, conf_threshold=0.1, nms_threshold=0.4):
#         """Run inference on a single sample from the full dataset."""
#         sample = self.dataset[sample_idx]
#         bev = sample['bev'].unsqueeze(0).to(self.device)
#         points = sample['points'].unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(bev, points)
#         return self._decode_outputs(outputs, sample, conf_threshold, nms_threshold)
    
#     def _decode_outputs(self, outputs, sample, conf_threshold, nms_threshold):
#         """Convert model outputs to usable bounding boxes."""
#         # Get predictions (shape: [max_objects, 7])
#         pred_boxes = outputs['reg_pred'][0].cpu().numpy()
#         # Apply softmax along the last dimension to get class probabilities
#         pred_cls = torch.softmax(outputs['cls_logits'], dim=2)[0].cpu().numpy()
        
#         # Filter by confidence threshold
#         conf_scores = np.max(pred_cls, axis=1)
#         valid_mask = conf_scores > conf_threshold
#         pred_boxes = pred_boxes[valid_mask]
#         pred_cls = pred_cls[valid_mask]
        
#         decoded_boxes = []
#         for box, cls_prob in zip(pred_boxes, pred_cls):
#             x, y, w, l, yaw, z, height = box
#             class_id = int(np.argmax(cls_prob))
#             decoded_boxes.append({
#                 'x': float(x),
#                 'y': float(y),
#                 'w': float(w),
#                 'l': float(l),
#                 'yaw': float(yaw),
#                 'z': float(z),
#                 'height': float(height),
#                 'class_id': class_id,
#                 'confidence': float(np.max(cls_prob))
#             })
        
#         # Apply NMS (convert center format [x, y, w, l] to corner format [x1, y1, x2, y2])
#         if decoded_boxes:
#             boxes_array = np.array([
#                 [b['x'] - b['w'] / 2, b['y'] - b['l'] / 2,
#                  b['x'] + b['w'] / 2, b['y'] + b['l'] / 2] for b in decoded_boxes
#             ])
#             scores = np.array([b['confidence'] for b in decoded_boxes])
#             boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32)
#             scores_tensor = torch.tensor(scores, dtype=torch.float32)
#             keep = nms(boxes_tensor, scores_tensor, nms_threshold)
#             keep = keep.cpu().numpy() if isinstance(keep, torch.Tensor) else keep
#             decoded_boxes = [decoded_boxes[i] for i in keep]
        
#         return decoded_boxes, sample
    
#     def visualize(self, pred_boxes, sample):
#         """Visualize predictions vs. ground truth on the BEV image."""
#         bev = sample['bev'].numpy().transpose(1, 2, 0)
#         plt.figure(figsize=(20, 10))
#         plt.subplot(121)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Predictions')
#         self._draw_boxes(pred_boxes, color='red')
#         plt.subplot(122)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Ground Truth')
#         self._draw_boxes(self._get_gt_boxes(sample), color='green')
#         plt.show()
    
#     def _draw_boxes(self, boxes, color):
#         """Helper to draw boxes on the BEV image."""
#         for box in boxes:
#             x = (box['x'] + (Config.bev_size[0]//2 * Config.bev_resolution)) / Config.bev_resolution
#             y = (box['y'] + (Config.bev_size[1]//2 * Config.bev_resolution)) / Config.bev_resolution
#             w = box['w'] / Config.bev_resolution
#             l = box['l'] / Config.bev_resolution
#             rect = plt.Rectangle((x - w/2, y - l/2), w, l,
#                                  angle=np.degrees(box['yaw']),
#                                  fill=False, color=color, linewidth=2)
#             plt.gca().add_patch(rect)
#             if 'confidence' in box:
#                 plt.text(x, y,
#                          f"{list(self.dataset.class_map.keys())[box['class_id']]}\n{box['confidence']:.2f}",
#                          color='white', fontsize=8, ha='center')
    
#     def _get_gt_boxes(self, sample):
#         """Convert ground truth boxes for visualization."""
#         gt_boxes = []
#         for box, label in zip(sample['target']['boxes'], sample['target']['labels']):
#             if label == -1:
#                 continue
#             gt_boxes.append({
#                 'x': float(box[0].item()),
#                 'y': float(box[1].item()),
#                 'w': float(box[2].item()),
#                 'l': float(box[3].item()),
#                 'yaw': float(box[4].item()),
#                 'z': float(box[5].item()),
#                 'height': float(box[6].item()),
#                 'class_id': int(label.item())
#             })
#         return gt_boxes
    
#     def evaluate(self, output_file='results.json'):
#         """
#         Run full evaluation using NuScenes metrics.
#         Only predictions for samples in the evaluation split will be kept.
#         """
#         results_dict = {}
#         # Build predictions for all samples in our dataset.
#         for sample_idx in tqdm(range(len(self.dataset)), desc="Generating Predictions"):
#             sample = self.dataset[sample_idx]
#             pred_boxes, _ = self.predict(sample_idx)
#             sample_token = self.nusc.sample[sample_idx]['token']
#             if sample_token not in results_dict:
#                 results_dict[sample_token] = []
#             # Convert predictions to the NuScenes result format.
#             for box in pred_boxes:
#                 raw_class_name = list(self.dataset.class_map.keys())[box['class_id']]
#                 mapped_name = self.detection_name_mapping.get(raw_class_name, raw_class_name)
#                 detection = {
#                     'sample_token': sample_token,
#                     'translation': [float(box['x']), float(box['y']), float(box['z'])],
#                     'size': [float(box['l']), float(box['w']), float(box['height'])],
#                     'rotation': [float(np.cos(box['yaw'] / 2.0)), 0, 0, float(np.sin(box['yaw'] / 2.0))],
#                     'detection_name': mapped_name,
#                     'detection_score': float(box['confidence']),
#                     'velocity': [0.0, 0.0],
#                     'attribute_name': ""
#                 }
#                 results_dict[sample_token].append(detection)
        
#         # --- Filter predictions to include only evaluation split samples ---
#         split_file = os.path.join(Config.dataroot, Config.version, 'split', f'{Config.eval_split}.txt')
#         if os.path.exists(split_file):
#             split_tokens = load_split_tokens(split_file)
#             # Only keep keys that are in the evaluation split.
#             results_dict = {token: results_dict.get(token, []) for token in split_tokens}
#         else:
#             print(f"Warning: Split file {split_file} not found. Using fallback mini_val scenes.")
#             # Fallback: define a set of scene names and extract all sample tokens from those scenes.
#             mini_val_scene_names = ['scene-0069', 'scene-0087', 'scene-0091', 'scene-0102']
#             mini_val_tokens = set()
#             for scene in self.nusc.scene:
#                 if scene['name'] in mini_val_scene_names:
#                     mini_val_tokens.update(get_scene_sample_tokens(self.nusc, scene['token']))
#             results_dict = {token: results_dict.get(token, []) for token in mini_val_tokens}
        
#         # Meta information required by NuScenes
#         meta = {
#             "use_camera": False,
#             "use_lidar": True,
#             "use_radar": False,
#             "use_map": False,
#             "use_external": False
#         }
#         final_results = {"results": results_dict, "meta": meta}
#         with open(output_file, 'w') as f:
#             json.dump(final_results, f)
        
#         # Run the official NuScenes evaluation.
#         nusc_eval = NuScenesEval(
#             self.nusc,
#             config=config_factory('detection_cvpr_2019'),
#             result_path=output_file,
#             eval_set=Config.eval_split,
#             output_dir='./eval_results'
#         )
#         return nusc_eval.main()

# if __name__ == '__main__':
#     evaluator = NuScenesEvaluator()
    
#     # Optionally visualize predictions on one sample.
#     pred_boxes, sample = evaluator.predict(0)
#     evaluator.visualize(pred_boxes, sample)
    
#     # Run full evaluation.
#     metrics = evaluator.evaluate()
#     print("Evaluation Metrics:")
#     print(metrics)

#-----------------------------------------------------------------------
#----------------------------------------------------------------------=

# import os
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from Model import (
#     Config,
#     NuScenesHybridDataset,
#     HybridDetector,
#     quaternion_yaw,
#     collate_fn
# )
# from nuscenes.nuscenes import NuScenes
# from tqdm import tqdm
# from torchvision.ops import nms

# # --- Set evaluation split, if you still want a particular split ---
# Config.eval_split = 'mini_val'

# class NuScenesVisualizer:
#     def __init__(self, model_path='hybrid_detector.pth'):
#         self.device = Config.device
#         self.nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
#         self.dataset = NuScenesHybridDataset(self.nusc)
        
#         # Load the trained model
#         self.model = HybridDetector(num_classes=Config.num_classes).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
        
#         # Mapping from dataset's raw class names to shorter labels for display
#         self.class_list = list(self.dataset.class_map.keys())

#     def predict(self, sample_idx, conf_threshold=0.1, nms_threshold=0.4):
#         """Run inference on a single sample from the dataset."""
#         sample = self.dataset[sample_idx]
#         bev = sample['bev'].unsqueeze(0).to(self.device)
#         points = sample['points'].unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(bev, points)
#         return self._decode_outputs(outputs, sample, conf_threshold, nms_threshold)

#     def _decode_outputs(self, outputs, sample, conf_threshold, nms_threshold):
#         """Convert model outputs to bounding boxes."""
#         # Get predictions (shape: [max_objects, 7])
#         pred_boxes = outputs['reg_pred'][0].cpu().numpy()
#         # Apply softmax to get class probabilities
#         pred_cls = torch.softmax(outputs['cls_logits'], dim=2)[0].cpu().numpy()
        
#         # Filter by confidence threshold
#         conf_scores = np.max(pred_cls, axis=1)
#         valid_mask = conf_scores > conf_threshold
#         pred_boxes = pred_boxes[valid_mask]
#         pred_cls = pred_cls[valid_mask]
        
#         decoded_boxes = []
#         for box, cls_prob in zip(pred_boxes, pred_cls):
#             x, y, w, l, yaw, z, height = box
#             class_id = int(np.argmax(cls_prob))
#             decoded_boxes.append({
#                 'x': float(x),
#                 'y': float(y),
#                 'w': float(w),
#                 'l': float(l),
#                 'yaw': float(yaw),
#                 'z': float(z),
#                 'height': float(height),
#                 'class_id': class_id,
#                 'confidence': float(np.max(cls_prob))
#             })
        
#         # Apply NMS to filter overlapping boxes
#         if decoded_boxes:
#             boxes_array = np.array([
#                 [b['x'] - b['w'] / 2, b['y'] - b['l'] / 2,
#                  b['x'] + b['w'] / 2, b['y'] + b['l'] / 2] 
#                 for b in decoded_boxes
#             ])
#             scores = np.array([b['confidence'] for b in decoded_boxes])
#             boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32)
#             scores_tensor = torch.tensor(scores, dtype=torch.float32)
#             keep = nms(boxes_tensor, scores_tensor, nms_threshold)
#             keep = keep.cpu().numpy() if isinstance(keep, torch.Tensor) else keep
#             decoded_boxes = [decoded_boxes[i] for i in keep]
        
#         return decoded_boxes, sample

#     def visualize_bev(self, pred_boxes, sample):
#         """Visualize predictions vs. ground truth on the BEV image."""
#         bev = sample['bev'].numpy().transpose(1, 2, 0)
#         plt.figure(figsize=(20, 10))
#         plt.subplot(121)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Predictions')
#         self._draw_boxes(pred_boxes, color='red')
        
#         plt.subplot(122)
#         plt.imshow(bev[..., 0], cmap='viridis')
#         plt.title('Ground Truth')
#         self._draw_boxes(self._get_gt_boxes(sample), color='green')
#         plt.show()

#     def _draw_boxes(self, boxes, color):
#         """Helper to draw boxes on the BEV image."""
#         for box in boxes:
#             x = (box['x'] + (Config.bev_size[0]//2 * Config.bev_resolution)) / Config.bev_resolution
#             y = (box['y'] + (Config.bev_size[1]//2 * Config.bev_resolution)) / Config.bev_resolution
#             w = box['w'] / Config.bev_resolution
#             l = box['l'] / Config.bev_resolution
            
#             rect = plt.Rectangle(
#                 (x - w/2, y - l/2),
#                 w, l,
#                 angle=np.degrees(box['yaw']),
#                 fill=False, color=color, linewidth=2
#             )
#             plt.gca().add_patch(rect)
            
#             if 'confidence' in box:
#                 class_name = self.class_list[box['class_id']]
#                 plt.text(x, y,
#                          f"{class_name}\n{box['confidence']:.2f}",
#                          color='white', fontsize=8, ha='center')

#     def _get_gt_boxes(self, sample):
#         """Convert ground truth boxes for visualization."""
#         gt_boxes = []
#         for box, label in zip(sample['target']['boxes'], sample['target']['labels']):
#             if label == -1:
#                 continue
#             gt_boxes.append({
#                 'x': float(box[0].item()),
#                 'y': float(box[1].item()),
#                 'w': float(box[2].item()),
#                 'l': float(box[3].item()),
#                 'yaw': float(box[4].item()),
#                 'z': float(box[5].item()),
#                 'height': float(box[6].item()),
#                 'class_id': int(label.item())
#             })
#         return gt_boxes

# if __name__ == '__main__':
#     visualizer = NuScenesVisualizer(model_path='hybrid_detector.pth')
    
#     # Predict and visualize on the 0th sample (you can pick any index).
#     pred_boxes, sample = visualizer.predict(sample_idx=0)
#     visualizer.visualize_bev(pred_boxes, sample)

#     # If you'd like to visualize multiple samples, do something like:
#     for i in range(5):
#         pred_boxes, sample = visualizer.predict(sample_idx=i)
#         visualizer.visualize_bev(pred_boxes, sample)


import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from Model import (
    Config,
    NuScenesHybridDataset,
    HybridDetector,
    quaternion_yaw,
    collate_fn
)
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from tqdm import tqdm
from torchvision.ops import nms

class NuScenesEvaluator:
    def __init__(self, model_path='hybrid_detector.pth'):
        self.device = Config.device
        self.nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)
        self.dataset = NuScenesHybridDataset(self.nusc)
        
        # Load the trained model
        self.model = HybridDetector(num_classes=Config.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Mapping from your dataset's raw class names to the 10 classes expected by NuScenes evaluation.
        self.detection_name_mapping = {
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.personal_mobility': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.barrier': 'barrier',
            'movable_object.debris': 'barrier',
            'movable_object.pushable_pullable': 'barrier',
            'movable_object.trafficcone': 'traffic_cone',
            'static_object.bicycle_rack': 'bicycle',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        
    def predict(self, sample_idx, conf_threshold=0.1, nms_threshold=0.4):
        """Run inference on a single sample."""
        sample = self.dataset[sample_idx]
        bev = sample['bev'].unsqueeze(0).to(self.device)
        points = sample['points'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(bev, points)
        
        return self._decode_outputs(outputs, sample, conf_threshold, nms_threshold)
    
    def _decode_outputs(self, outputs, sample, conf_threshold, nms_threshold):
        """Convert model outputs to usable bounding boxes."""
        # Get predictions (shape: [max_objects, 7])
        pred_boxes = outputs['reg_pred'][0].cpu().numpy()
        # Apply softmax along the last dimension to get class probabilities
        pred_cls = torch.softmax(outputs['cls_logits'], dim=2)[0].cpu().numpy()  # Shape: [max_objects, num_classes]
        
        # Filter predictions by confidence threshold
        conf_scores = np.max(pred_cls, axis=1)
        valid_mask = conf_scores > conf_threshold
        
        pred_boxes = pred_boxes[valid_mask]
        pred_cls = pred_cls[valid_mask]
        
        # Decode boxes into a list of dictionaries
        decoded_boxes = []
        for box, cls_prob in zip(pred_boxes, pred_cls):
            x, y, w, l, yaw, z, height = box
            class_id = int(np.argmax(cls_prob))
            decoded_boxes.append({
                'x': float(x),  # x and y are assumed to be in BEV coordinate space
                'y': float(y),
                'w': float(w),
                'l': float(l),
                'yaw': float(yaw),
                'z': float(z),
                'height': float(height),
                'class_id': class_id,
                'confidence': float(np.max(cls_prob))
            })
        
        # Apply NMS: convert from center format [x, y, w, l] to corner format [x1, y1, x2, y2]
        if decoded_boxes:
            boxes_array = np.array([
                [
                    b['x'] - b['w'] / 2,
                    b['y'] - b['l'] / 2,
                    b['x'] + b['w'] / 2,
                    b['y'] + b['l'] / 2
                ]
                for b in decoded_boxes
            ])
            scores = np.array([b['confidence'] for b in decoded_boxes])
            boxes_tensor = torch.tensor(boxes_array, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            keep = nms(boxes_tensor, scores_tensor, nms_threshold)
            # Convert to numpy array for indexing if needed
            keep = keep.cpu().numpy() if isinstance(keep, torch.Tensor) else keep
            decoded_boxes = [decoded_boxes[i] for i in keep]
        
        return decoded_boxes, sample
    
    def visualize(self, pred_boxes, sample):
        """Visualize predictions vs ground truth on the BEV image."""
        bev = sample['bev'].numpy().transpose(1, 2, 0)
        
        plt.figure(figsize=(20, 10))
        
        # Plot predictions
        plt.subplot(121)
        plt.imshow(bev[..., 0], cmap='viridis')
        plt.title('Predictions')
        self._draw_boxes(pred_boxes, color='red')
        
        # Plot ground truth
        plt.subplot(122)
        plt.imshow(bev[..., 0], cmap='viridis')
        plt.title('Ground Truth')
        self._draw_boxes(self._get_gt_boxes(sample), color='green')
        
        plt.show()
    
    def _draw_boxes(self, boxes, color):
        """Helper function to draw boxes on the BEV image."""
        for box in boxes:
            # Convert BEV coordinates to pixel coordinates
            x = (box['x'] + (Config.bev_size[0]//2 * Config.bev_resolution)) / Config.bev_resolution
            y = (box['y'] + (Config.bev_size[1]//2 * Config.bev_resolution)) / Config.bev_resolution
            w = box['w'] / Config.bev_resolution
            l = box['l'] / Config.bev_resolution
            
            rect = plt.Rectangle(
                (x - w/2, y - l/2), w, l,
                angle=np.degrees(box['yaw']),
                fill=False,
                color=color,
                linewidth=2
            )
            plt.gca().add_patch(rect)
            if 'confidence' in box:
                plt.text(x, y, 
                         f"{list(self.dataset.class_map.keys())[box['class_id']]}\n{box['confidence']:.2f}",
                         color='white', fontsize=8, ha='center')
    
    def _get_gt_boxes(self, sample):
        """Convert ground truth boxes to visualization format."""
        gt_boxes = []
        for box, label in zip(sample['target']['boxes'], sample['target']['labels']):
            if label == -1:
                continue
            gt_boxes.append({
                'x': float(box[0].item()),
                'y': float(box[1].item()),
                'w': float(box[2].item()),
                'l': float(box[3].item()),
                'yaw': float(box[4].item()),
                'z': float(box[5].item()),
                'height': float(box[6].item()),
                'class_id': int(label.item())
            })
        return gt_boxes
    
    def evaluate(self, output_file='results.json'):
        """
        Run full evaluation using NuScenes metrics.
        The results are reformatted into a dictionary with top-level 'results' and 'meta' fields.
        """
        results_dict = {}
        
        for sample_idx in tqdm(range(len(self.dataset))):
            sample = self.dataset[sample_idx]
            pred_boxes, _ = self.predict(sample_idx)
            sample_token = self.nusc.sample[sample_idx]['token']
            
            # Initialize list for this sample if not already
            if sample_token not in results_dict:
                results_dict[sample_token] = []
            
            # Convert predictions to NuScenes result format.
            # Use the detection_name_mapping to map the raw class name to the expected one.
            for box in pred_boxes:
                # Get the original class name from your dataset's class_map.
                raw_class_name = list(self.dataset.class_map.keys())[box['class_id']]
                mapped_name = self.detection_name_mapping.get(raw_class_name, raw_class_name)
                
                detection = {
                    'sample_token': sample_token,
                    'translation': [float(box['x']), float(box['y']), float(box['z'])],
                    'size': [float(box['l']), float(box['w']), float(box['height'])],
                    'rotation': [float(np.cos(box['yaw'] / 2.0)), 0, 0, float(np.sin(box['yaw'] / 2.0))],
                    'detection_name': mapped_name,
                    'detection_score': float(box['confidence']),
                    'velocity': [0.0, 0.0],  # Default velocity
                    'attribute_name': ""     # Default attribute
                }
                results_dict[sample_token].append(detection)
        
        # Meta information required by NuScenes
        meta = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        }
        
        final_results = {"results": results_dict, "meta": meta}
        
        # Save the results in the required format
        with open(output_file, 'w') as f:
            json.dump(final_results, f)
            
        # Run the official NuScenes evaluation
        nusc_eval = NuScenesEval(
            self.nusc,
            config=config_factory('detection_cvpr_2019'),
            result_path=output_file,
            eval_set='mini_val',
            output_dir='./eval_results'
        )
        return nusc_eval.main()

if __name__ == '__main__':
    evaluator = NuScenesEvaluator()
    
    # Example: Visualize predictions on sample 0
    pred_boxes, sample = evaluator.predict(0)
    evaluator.visualize(pred_boxes, sample)
    
    # Full evaluation
    metrics = evaluator.evaluate()
    print("Evaluation Metrics:")
    print(metrics)
