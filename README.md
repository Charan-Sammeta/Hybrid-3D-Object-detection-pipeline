# Hybrid 3D Object Detection Pipeline

This repository implements a state-of-the-art Hybrid 3D Object Detection approach, leveraging both Bird's Eye View (BEV) images and direct point cloud analysis to enhance object detection accuracy.

## Project Structure
```
NUSCENES
│-- 3D-Detection-Pipeline
│   │-- Code
│   │   │-- bin_to_pcd.py         # Converts bin files to PCD format
│   │   │-- BoundingBoxes.py      # Generates bounding boxes for detected clusters
│   │   │-- Clustering.py         # Clusters segmented point cloud data
│   │   │-- Detection_Pipeline.py # Main detection pipeline script
│   │   │-- Downsampling.py       # Downsamples point cloud for efficiency
│   │   │-- Segmentation.py       # Segments objects from the point cloud
│   │   │-- Video_Pipeline.py     # Processes and visualizes pipeline output
│   │-- Data
│   │-- Results
│-- eval_results
│-- maps
│-- samples
│-- sweeps
│-- v1.0-mini
│-- convert.py          # Converts input data into .pcd files
│-- evaluate.py         # Evaluates the detection model
│-- load_data.py        # Loads and processes input data
│-- test.pcd            # Sample point cloud data
│-- results.json        # Stores detection results
│-- LICENSE
│-- README.md
```

## Overview
This project implements a **Hybrid 3D Object Detection** system that combines BEV representations derived from lidar point clouds with raw point cloud analysis. By leveraging both structured spatial grids and raw point-based features, the model enhances object detection accuracy.

### Configurations and Initial Setup
The system is built around a **Config class** that defines crucial parameters:
- **Paths and Versions:** Specifies paths for dataset storage and versions.
- **Data Parameters:** Determines the BEV resolution, impacting 3D representation in 2D.
- **Model Parameters:** Defines feature dimensions and network architecture.
- **Training Parameters:** Includes batch size, learning rate, and epoch count.

### Dataset Handling and BEV Image Generation
The **NuScenesHybridDataset** class processes the dataset and converts complex 3D data into a structured 2D format for CNNs.
- **Class Mapping:** Dynamically categorizes objects.
- **BEV Transformation:** Converts raw lidar data into a 2D BEV format while retaining crucial spatial information.
- **Data Preparation:** Prepares BEV tensors, raw point cloud data, bounding boxes, and class labels for training.

## Model Architecture and Fusion Strategy
The hybrid model consists of two main components:
1. **BEVNet:** A CNN that extracts spatial features from BEV images, identifying environmental structure.
2. **PointNetBranch:** A PointNet-based model that processes raw point cloud data, capturing detailed point-level features.
3. **Feature Fusion & Detection Output:** Features from both branches are merged using fully connected layers, and separate heads process bounding box regression and classification.

## Loss Computation and Training Pipeline
The **HybridLoss** class computes:
- **Classification Loss:** Cross-entropy loss.
- **Localization Loss:** Smooth L1 loss.

The training loop:
- Iterates over data batches.
- Compares predictions to true labels.
- Uses computed loss to update model weights via backpropagation.

## Evaluation and Visualization Tools
The **NuScenesEvaluator** class assesses model performance:
1. **Prediction & Decoding:** Bounding boxes and class probabilities are processed using confidence thresholding and Non-Maximum Suppression (NMS).
2. **Visualization:** Model predictions are displayed alongside ground truth annotations in BEV form.
3. **Evaluation Metrics:** Performance is benchmarked against NuScenes evaluation tools.

## Visualization Explanation
- The visualization presents **model predictions vs. ground truth annotations** in a BEV scene.
- BEV projection simplifies 3D object detection by converting point clouds into a structured 2D grid.
- The results are stored in `results.json`.

## Extending to 3D Cuboid Detection
For further **3D cuboid detection**, the predictions can be mapped to the point cloud to generate bounding boxes. This can be integrated with frameworks like **mmdetection** or **OpenPCDet**, which require an **Nvidia GPU (>=3060 Ti)** with **CUDA and CuDNN** support.

## Installation & Setup
### Prerequisites
Ensure the following dependencies are installed:
- Python 3.7+
- Open3D
- NumPy
- OpenCV
- PyTorch

### Setup Virtual Environment (Optional but Recommended)
```sh
python -m venv nuscenes_env
source nuscenes_env/bin/activate  # Windows: nuscenes_env\Scripts\activate
pip install -r requirements.txt
```

### Clone the Repository
```sh
git clone <repo_url>
cd nuscenes
```

## Running the Pipeline
### Convert Data to PCD Format
```sh
python convert.py
```

### Run the 3D Object Detection Pipeline
```sh
python 3D-Detection-Pipeline/Code/Detection_Pipeline.py
```

### Expected Output
```
115151
Downsample Time 0.04256939888000488
Segmentation Time 0.046744346618652344
[Open3D DEBUG] Precompute neighbors.
[Open3D DEBUG] Done Precompute neighbors.
[Open3D DEBUG] Compute Clusters
[Open3D DEBUG] Done Compute Clusters: 160
point cloud has 160 clusters
Clustering Time 0.08386087417602539
Number of Bounding Boxes calculated 44
Bounding Boxes Time 0.04169154167175293
```

The bounding boxes and detected objects will be displayed visually.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contributions
Feel free to contribute by submitting pull requests or reporting issues.

## Contact
For questions, please open an issue in the repository.

