# FinalLasClassifier

A comprehensive Python toolkit for LiDAR point cloud classification and processing using deep learning and traditional machine learning methods. This project provides tools for clustering, classification, generalization, and merging of LAS point cloud files.

## Features

-   **Point Cloud Classification**: Deep learning-based classification using PointNet architecture
-   **Spatial Clustering**: Density-based clustering (DBSCAN) and K-means clustering for object extraction
-   **Point Cloud Generalization**: Multiple downsampling methods (random, percentage, grid-based)
-   **Terrain Classification**: K-means based terrain classification using geometric and attribute features
-   **File Merging**: Efficient merging of multiple LAS files while preserving point positions
-   **Automated Pipeline**: Script execution system for batch processing

## Project Structure

```
FinalLasClassifier/
├── point_cloud_classifier.py    # Main classification module with PointNet architecture
├── las_color_cluster.py          # Spatial density-based clustering
├── las_color_cluster2.py         # Optimized clustering with item identification
├── las_flooring_divide.py        # Floor/terrain point separation
├── las_generalize.py             # Point cloud downsampling/generalization
├── clustring_floor.py            # Terrain classification using K-means
├── Combine.py                    # Merge multiple LAS files
├── executeFiles.py               # Automated pipeline execution
├── chwilats.pth                  # Trained model weights
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd FinalLasClassifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

-   Python 3.7+
-   PyTorch (with CUDA support recommended for GPU acceleration)
-   NumPy
-   laspy
-   scikit-learn

See `requirements.txt` for specific versions.

## Usage

### 1. Point Cloud Classification

Classify point clouds using a trained PointNet model:

```bash
python point_cloud_classifier.py
```

The script will:

-   Load the trained model from `chwilats.pth`
-   Classify all `.las` files in the `Clusters/` directory
-   Update classification values in the LAS files
-   Display classification summary

**Class Labels:**

-   0: trees
-   1: trampoles
-   2: roads
-   3: rails
-   4: columns
-   5: none

### 2. Spatial Clustering

Extract objects from point clouds using density-based clustering:

```bash
# DBSCAN clustering (recommended for object extraction)
python las_color_cluster2.py input.las

# With custom parameters
python las_color_cluster2.py input.las --eps-spatial 2.5 --min-samples 200 --max-points 200000
```

**Options:**

-   `--eps-spatial VALUE`: Maximum spatial distance for DBSCAN (default: 2.5)
-   `--min-samples VALUE`: Minimum points per cluster (default: 200)
-   `--max-points VALUE`: Maximum points to process (default: 200000)
-   `--method METHOD`: Clustering method ('dbscan' or 'minibatch')
-   `--output-dir DIR`: Output directory (default: 'Clusters')

### 3. Floor/Terrain Separation

Separate floor/terrain points from remaining points:

```bash
python las_flooring_divide.py input.las
```

This creates two output files:

-   `Objects/Class2_points.las`: Floor/terrain points
-   `Objects/Remaining_points.las`: Remaining points

### 4. Point Cloud Generalization

Downsample point clouds using various methods:

```bash
# Random sampling
python las_generalize.py data.las random 1000000

# Percentage sampling
python las_generalize.py data.las percentage 10

# Grid-based downsampling
python las_generalize.py data.las grid 1.0
```

### 5. Terrain Classification

Classify terrain points into clusters (e.g., grass, road, rails):

```bash
python clustring_floor.py
```

Uses K-means clustering based on:

-   Height (Z coordinate)
-   Intensity
-   Color (RGB)
-   Local variance (smoothness)

### 6. Merge LAS Files

Combine multiple LAS files into a single file:

```bash
python Combine.py
```

Merges all `.las` files from the `Clusters/` directory into `Merged_Points.las`.

### 7. Automated Pipeline

Execute the complete processing pipeline:

```bash
python executeFiles.py
```

This runs the following steps in sequence:

1. Floor/terrain separation
2. Clustering of remaining points
3. Point cloud classification
4. File merging

## Model Training

To train a new PointNet classifier:

```python
from point_cloud_classifier import PointNetClassifier, PointCloudClassifierTrainer, load_dataset
from torch.utils.data import DataLoader, Dataset

# Load your dataset
las_files = ["path/to/file1.las", "path/to/file2.las", ...]
labels = [0, 1, 2, ...]  # Class labels
train_data, train_labels = load_dataset(las_files, labels, num_points=2048)

# Create data loaders
train_dataset = PointCloudDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize and train model
model = PointNetClassifier(num_points=2048, num_classes=6)
trainer = PointCloudClassifierTrainer(model)
trainer.train(train_loader, epochs=25, learning_rate=0.01, save_path='model.pth')
```

## Configuration

### Model Parameters

The default model configuration:

-   **num_points**: 2048 (number of points per point cloud)
-   **num_classes**: 6 (trees, trampoles, roads, rails, columns, none)
-   **Model file**: `chwilats.pth`

### Clustering Parameters

Default clustering settings:

-   **eps_spatial**: 2.5 (spatial distance threshold)
-   **min_samples**: 200 (minimum points per cluster)
-   **max_points**: 200000 (maximum points to process)

## Output Directories

The project uses the following directory structure:

-   `Clusters/`: Contains clustered point cloud files
-   `Objects/`: Contains separated floor and remaining points
-   `Dataset/`: Training dataset (if applicable)

## Performance Notes

-   **GPU Acceleration**: PyTorch will automatically use CUDA if available
-   **Memory Management**: Large point clouds are automatically downsampled
-   **Clustering**: DBSCAN is recommended for object extraction; MiniBatch K-means for very large datasets

## Troubleshooting

### Memory Issues

-   Reduce `--max-points` parameter for clustering
-   Use grid-based downsampling before clustering
-   Process files in smaller batches

### Model Loading Errors

-   Ensure `chwilats.pth` exists in the project directory
-   Verify model parameters match training configuration (num_points, num_classes)

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

-   PointNet architecture for point cloud classification
-   laspy for LAS file handling
-   scikit-learn for clustering algorithms
