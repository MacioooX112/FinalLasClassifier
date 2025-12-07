"""
Optimized script to find clusters in LiDAR point cloud based on spatial density.
Dense regions (objects) are separated from sparse regions (empty space).
Each cluster is exported to a separate .las file with item identification features.
"""

import sys
import os
import numpy as np

try:
    import laspy
except ImportError:
    print("Error: laspy library is not installed.")
    print("Please install it using: pip install laspy")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN, MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install it for better clustering: pip install scikit-learn")


def cluster_by_density(las_file_path, n_clusters=50, method='dbscan', eps_spatial=1.0, min_samples=10, 
                       output_dir="Clusters", max_points=1000000):
    """
    Optimized cluster points based on spatial density for object extraction.
    
    Args:
        las_file_path (str): Path to the input .las file
        n_clusters (int): Number of clusters for K-means (default: 50)
        method (str): Clustering method ('dbscan' for density, 'minibatch')
        eps_spatial (float): Max spatial distance for DBSCAN (default: 1.0)
        min_samples (int): Minimum points per cluster for DBSCAN (default: 10)
        output_dir (str): Directory to save output files
        max_points (int): Maximum points to process (default: 1M)
        
    Returns:
        list: List of output file paths
    """
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"File not found: {las_file_path}")
    
    if not las_file_path.lower().endswith('.las'):
        raise ValueError(f"File must be a .las file: {las_file_path}")
    
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    # Get coordinates as single array (more memory efficient)
    # Use float32 to save memory (sufficient precision for most LiDAR data)
    coords = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    total_points = len(coords)
    print(f"  Total points: {total_points:,}")
    
    # Smart downsampling: use spatial grid for better representation
    if total_points > max_points:
        print(f"  Downsampling from {total_points:,} to {max_points:,} points...")
        # Use grid-based downsampling for better spatial distribution
        coords, sample_indices = grid_downsample(coords, max_points)
        original_indices = sample_indices
    else:
        original_indices = np.arange(total_points, dtype=np.uint32)
    
    num_points = len(coords)
    print(f"  Processing {num_points:,} points...")
    
    # Normalize coordinates efficiently (single pass)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1.0  # Avoid division by zero
    
    print(f"  Spatial range: X={coords_range[0]:.2f}, Y={coords_range[1]:.2f}, Z={coords_range[2]:.2f}")
    
    # Normalize in-place to save memory
    coords_norm = (coords - coords_min) / coords_range
    
    # Choose clustering method
    print(f"  Clustering method: {method.upper()}")
    
    if SKLEARN_AVAILABLE:
        if method.lower() == 'dbscan':
            # Optimized DBSCAN with adaptive parameters
            eps_normalized = eps_spatial / coords_range.max()
            print(f"  Using optimized DBSCAN (eps={eps_spatial:.2f} -> {eps_normalized:.4f} normalized, min_samples={min_samples})...")
            
            # Use optimized DBSCAN with memory-efficient settings
            cluster_labels = optimized_dbscan(coords_norm, eps_normalized, min_samples, num_points)
            print(f"  [OK] DBSCAN clustering complete!")
        
        elif method.lower() == 'minibatch':
            # MiniBatch K-means: Very fast and memory efficient
            print(f"  Using MiniBatchKMeans with {n_clusters} clusters...")
            batch_size = min(10000, max(1000, num_points // 20))
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                batch_size=batch_size, 
                n_init=3,
                max_iter=100
            )
            cluster_labels = clusterer.fit_predict(coords_norm)
            print(f"  [OK] MiniBatchKMeans clustering complete!")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dbscan' or 'minibatch'")
    else:
        # Simple fallback clustering
        print("  Using simple distance-based clustering...")
        eps_normalized = eps_spatial / coords_range.max()
        cluster_labels = simple_clustering(coords_norm, eps_normalized, min_samples)
    
    # Get unique clusters (excluding noise -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    num_clusters = len(unique_clusters)
    noise_points = np.sum(cluster_labels == -1) if method.lower() == 'dbscan' else 0
    
    print(f"  Found {num_clusters} clusters")
    if noise_points > 0:
        print(f"  Noise points: {noise_points:,}")
    
    # Setup output directory
    if output_dir is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_dir = os.path.dirname(base_name) if os.path.dirname(base_name) else '.'
        base_name = os.path.basename(base_name)
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(las_file_path))[0]
    
    # Export clusters with item identification features
    output_files = []
    print(f"\nExporting clusters with item identification...")
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = np.sum(cluster_mask)
        
        # Skip very small clusters
        if method.lower() != 'minibatch' and cluster_points < min_samples:
            continue
        
        # Get original indices for this cluster
        cluster_original_indices = original_indices[cluster_mask]
        
        # Calculate item identification features
        cluster_coords = coords[cluster_mask]
        item_features = calculate_item_features(cluster_coords, cluster_points)
        
        # Create output filename with item info
        item_type = identify_item_type(item_features)
        output_filename = f"{base_name}_cluster_{cluster_id:04d}_{cluster_points}pts_{item_type}.las"
        output_path = os.path.join(output_dir, output_filename)
        
        # Efficiently create LAS file
        create_cluster_las(las, cluster_original_indices, output_path)
        output_files.append(output_path)
        
        print(f"  Cluster {cluster_id}: {cluster_points:,} pts | {item_type} | "
              f"Size: {item_features['size']:.2f} | Density: {item_features['density']:.1f}")
    
    print(f"\n[OK] Successfully created {len(output_files)} cluster files!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    
    return output_files


def grid_downsample(coords, target_points):
    """
    Efficient grid-based downsampling for better spatial distribution.
    """
    # Calculate grid size
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    
    # Estimate grid resolution
    volume = np.prod(coords_range)
    if volume > 0:
        cell_size = (volume / target_points) ** (1/3)
        grid_res = np.maximum(1, (coords_range / cell_size).astype(int))
    else:
        grid_res = np.array([10, 10, 10], dtype=int)
    
    # Create grid indices
    grid_indices = ((coords - coords_min) / (coords_range / grid_res)).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_res - 1)
    
    # Hash grid cells
    grid_hash = (grid_indices[:, 0] * grid_res[1] * grid_res[2] + 
                 grid_indices[:, 1] * grid_res[2] + 
                 grid_indices[:, 2])
    
    # Sample one point per grid cell
    unique_cells, indices = np.unique(grid_hash, return_index=True)
    
    # If we need more points, randomly sample from remaining
    if len(indices) < target_points:
        remaining = np.setdiff1d(np.arange(len(coords)), indices)
        if len(remaining) > 0:
            additional = np.random.choice(remaining, 
                                        min(len(remaining), target_points - len(indices)), 
                                        replace=False)
            indices = np.concatenate([indices, additional])
    
    # Limit to target
    if len(indices) > target_points:
        indices = np.random.choice(indices, target_points, replace=False)
    
    return coords[indices], indices


def optimized_dbscan(coords_norm, eps, min_samples, num_points):
    """
    Memory-optimized DBSCAN implementation.
    """
    # For very large datasets, use chunked approach or reduce further
    if num_points > 300000:
        # Use adaptive min_samples for large datasets
        adaptive_min_samples = max(min_samples, int(min_samples * 0.5))
        print(f"  Using adaptive min_samples={adaptive_min_samples} for large dataset...")
    else:
        adaptive_min_samples = min_samples
    
    # Use optimized DBSCAN parameters
    clusterer = DBSCAN(
        eps=eps,
        min_samples=adaptive_min_samples,
        metric='euclidean',
        n_jobs=1,  # Single-threaded to avoid memory issues
        algorithm='auto'  # Let sklearn choose best algorithm
    )
    
    try:
        cluster_labels = clusterer.fit_predict(coords_norm)
    except MemoryError:
        # If still fails, reduce dataset further
        print(f"  Memory error, reducing dataset size...")
        reduce_to = min(200000, num_points // 2)
        sample_indices = np.random.choice(num_points, reduce_to, replace=False)
        coords_reduced = coords_norm[sample_indices]
        cluster_labels_reduced = clusterer.fit_predict(coords_reduced)
        
        # Map back to full dataset (assign remaining points to nearest cluster)
        print(f"  Mapping clusters back to full dataset...")
        cluster_labels = map_clusters_to_full(coords_norm, coords_reduced, 
                                            cluster_labels_reduced, sample_indices, eps)
    
    return cluster_labels


def map_clusters_to_full(coords_full, coords_sampled, labels_sampled, sample_indices, eps):
    """
    Map cluster labels from sampled points back to full dataset.
    """
    # Find nearest neighbor for each unsampled point
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=1)
    nn.fit(coords_sampled)
    
    # Get labels for all points
    full_labels = np.full(len(coords_full), -1, dtype=np.int32)
    full_labels[sample_indices] = labels_sampled
    
    # For unsampled points, assign to nearest cluster if within eps
    unsampled_mask = np.ones(len(coords_full), dtype=bool)
    unsampled_mask[sample_indices] = False
    unsampled_indices = np.where(unsampled_mask)[0]
    
    if len(unsampled_indices) > 0:
        distances, nearest = nn.kneighbors(coords_full[unsampled_indices])
        within_eps = distances[:, 0] <= eps
        valid_nearest = nearest[within_eps, 0]
        full_labels[unsampled_indices[within_eps]] = labels_sampled[valid_nearest]
    
    return full_labels


def calculate_item_features(cluster_coords, num_points):
    """
    Calculate features to help identify item type.
    """
    if num_points < 3:
        return {'size': 0, 'density': 0, 'aspect_ratio': 1.0, 'height': 0, 'volume': 0}
    
    # Size (bounding box diagonal)
    bbox_min = cluster_coords.min(axis=0)
    bbox_max = cluster_coords.max(axis=0)
    bbox_size = bbox_max - bbox_min
    size = np.linalg.norm(bbox_size)
    
    # Volume (approximate)
    volume = np.prod(bbox_size) if np.all(bbox_size > 0) else 0
    
    # Density (points per unit volume)
    density = num_points / (volume + 0.001)
    
    # Aspect ratio (how elongated)
    sorted_sizes = np.sort(bbox_size)
    aspect_ratio = sorted_sizes[2] / (sorted_sizes[0] + 0.001) if sorted_sizes[0] > 0 else 1.0
    
    # Height (Z dimension)
    height = bbox_size[2]
    
    return {
        'size': size,
        'density': density,
        'aspect_ratio': aspect_ratio,
        'height': height,
        'volume': volume
    }


def identify_item_type(features):
    """
    Identify item type based on features.
    """
    size = features['size']
    density = features['density']
    aspect_ratio = features['aspect_ratio']
    height = features['height']
    
    # Classification rules
    if size < 0.5:
        return "small"
    elif size < 2.0:
        if aspect_ratio > 3.0:
            return "pole"
        elif height > 1.5:
            return "tall"
        elif density > 1000:
            return "dense"
        else:
            return "medium"
    elif size < 5.0:
        if aspect_ratio < 1.5:
            return "box"
        else:
            return "large"
    else:
        return "very_large"


def create_cluster_las(las, cluster_indices, output_path):
    """
    Efficiently create LAS file for a cluster.
    """
    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header.offsets = las.header.offsets
    out_las.header.scales = las.header.scales
    
    # Copy points efficiently
    out_las.points = las.points[cluster_indices]
    
    # Update bounds efficiently
    out_las.header.x_min = float(out_las.x.min())
    out_las.header.x_max = float(out_las.x.max())
    out_las.header.y_min = float(out_las.y.min())
    out_las.header.y_max = float(out_las.y.max())
    out_las.header.z_min = float(out_las.z.min())
    out_las.header.z_max = float(out_las.z.max())
    
    out_las.write(output_path)


def simple_clustering(features, eps, min_samples):
    """
    Simple distance-based clustering (fallback).
    """
    n_points = len(features)
    cluster_labels = np.full(n_points, -1, dtype=np.int32)
    cluster_id = 0
    
    for i in range(n_points):
        if cluster_labels[i] != -1:
            continue
        
        # Vectorized distance calculation
        distances = np.linalg.norm(features - features[i], axis=1)
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) >= min_samples:
            cluster_labels[neighbors] = cluster_id
            cluster_id += 1
    
    return cluster_labels


def main():
    """Main function to run the script."""
    if len(sys.argv) < 2:
        print("Usage: python las_color_cluster.py <path_to_las_file> [OPTIONS]")
        print("\nOptions:")
        print("  --n-clusters VALUE      Number of clusters for MiniBatch (default: 50)")
        print("  --method METHOD         Clustering method: 'dbscan' (default) or 'minibatch'")
        print("  --eps-spatial VALUE     Max spatial distance for DBSCAN (default: 1.0)")
        print("  --min-samples VALUE     Minimum points per cluster for DBSCAN (default: 10)")
        print("  --output-dir DIR        Output directory (default: 'Clusters')")
        print("  --max-points VALUE      Max points to process (default: 1000000)")
        print("\nExamples:")
        print("  # Density-based clustering with DBSCAN (recommended)")
        print("  python las_color_cluster.py data.las")
        print("  # Tighter clustering (smaller objects)")
        print("  python las_color_cluster.py data.las --eps-spatial 0.5")
        print("  # MiniBatch for very large files")
        print("  python las_color_cluster.py data.las --method minibatch --n-clusters 100")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    
    # Default parameters (optimized for 500k+ points)
    n_clusters = 50
    method = 'dbscan'
    eps_spatial = 2.5
    min_samples = 200
    output_dir = "Clusters"
    max_points = 200000  # Increased default
    
    # Parse arguments
    if '--n-clusters' in sys.argv:
        idx = sys.argv.index('--n-clusters')
        if idx + 1 < len(sys.argv):
            n_clusters = int(sys.argv[idx + 1])
    
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1].lower()
    
    if '--eps-spatial' in sys.argv:
        idx = sys.argv.index('--eps-spatial')
        if idx + 1 < len(sys.argv):
            eps_spatial = float(sys.argv[idx + 1])
    
    if '--min-samples' in sys.argv:
        idx = sys.argv.index('--min-samples')
        if idx + 1 < len(sys.argv):
            min_samples = int(sys.argv[idx + 1])
    
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    if '--max-points' in sys.argv:
        idx = sys.argv.index('--max-points')
        if idx + 1 < len(sys.argv):
            max_points = int(sys.argv[idx + 1])
    
    try:
        output_files = cluster_by_density(
            las_file_path,
            n_clusters=n_clusters,
            method=method,
            eps_spatial=eps_spatial,
            min_samples=min_samples,
            output_dir=output_dir,
            max_points=max_points
        )
        print(f"\nAll done! Created {len(output_files)} cluster files.")
        return output_files
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
