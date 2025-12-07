"""
Script to find clusters in LiDAR point cloud based on spatial density.
Dense regions (objects) are separated from sparse regions (empty space).
Each cluster is exported to a separate .las file.
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
    from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install it for better clustering: pip install scikit-learn")


def calculate_color_distance(rgb1, rgb2):
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        rgb1: First RGB color (array or tuple)
        rgb2: Second RGB color (array or tuple)
        
    Returns:
        float: Color distance
    """
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))


def cluster_by_density(las_file_path, n_clusters=50, method='dbscan', eps_spatial=1.0, min_samples=10, 
                       output_dir=None, max_points=10000000):
    """
    Cluster points based on spatial density for object extraction.
    Dense regions (objects) are separated from sparse regions (empty space).
    
    Args:
        las_file_path (str): Path to the input .las file
        n_clusters (int): Number of clusters to create (for K-means, default: 50)
        method (str): Clustering method ('dbscan' for density, 'kmeans', or 'minibatch')
        eps_spatial (float): Maximum spatial distance for density clustering (default: 1.0)
        min_samples (int): Minimum points per cluster (for DBSCAN, default: 10)
        output_dir (str): Directory to save output files (optional)
        max_points (int): Maximum points to process (for large files)
        
    Returns:
        list: List of output file paths
    """
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"File not found: {las_file_path}")
    
    if not las_file_path.lower().endswith('.las'):
        raise ValueError(f"File must be a .las file: {las_file_path}")
    
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    # Get coordinates
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)
    z_coords = np.array(las.z)
    
    total_points = len(x_coords)
    print(f"   Total points: {total_points:,}")
    
    # For density-based clustering, we only need spatial coordinates
    print(f"   Using spatial density-based clustering (X, Y, Z coordinates)")
    
    # Downsample if too many points
    if total_points > max_points:
        print(f"   Downsampling from {total_points:,} to {max_points:,} points for clustering...")
        sample_indices = np.random.choice(total_points, max_points, replace=False)
        x_coords = x_coords[sample_indices]
        y_coords = y_coords[sample_indices]
        z_coords = z_coords[sample_indices]
        original_indices = sample_indices
    else:
        original_indices = np.arange(total_points)
    
    num_points = len(x_coords)
    print(f"   Processing {num_points:,} points...")
    
    # Normalize spatial coordinates for clustering
    # This ensures all dimensions are on similar scales
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    z_range = z_max - z_min if z_max != z_min else 1.0
    
    print(f"   Spatial range: X={x_range:.2f}, Y={y_range:.2f}, Z={z_range:.2f}")
    
    # Normalize coordinates to 0-1 range for better clustering
    x_norm = (x_coords - x_min) / x_range
    y_norm = (y_coords - y_min) / y_range
    z_norm = (z_coords - z_min) / z_range
    
    # Create feature matrix: spatial coordinates only (for density clustering)
    print(f"   Creating spatial density feature matrix (X, Y, Z normalized)...")
    
    # Use only spatial coordinates for density-based clustering
    features = np.column_stack([
        x_norm,
        y_norm,
        z_norm
    ])
    
    # Choose clustering method
    print(f"   Clustering method: {method.upper()}")
    
    if SKLEARN_AVAILABLE:
        if method.lower() == 'dbscan':
            # DBSCAN: Best for density-based clustering - identifies dense regions (objects)
            # Normalize eps_spatial to the normalized coordinate space (0-1 range)
            # eps_spatial is in original units, convert to normalized space
            eps_normalized = eps_spatial / max(x_range, y_range, z_range)
            print(f"   Using DBSCAN density clustering (eps={eps_spatial:.2f} units -> {eps_normalized:.4f} normalized, min_samples={min_samples})...")
            print(f"   This will identify dense regions (objects) vs sparse regions (empty space)...")
            
            n_jobs = 1  # Always use single-threaded to avoid memory issues
            if num_points > 100000:
                print(f"   Using single-threaded mode for memory efficiency...")
            
            try:
                clusterer = DBSCAN(eps=eps_normalized, min_samples=min_samples, metric='euclidean', n_jobs=n_jobs)
                cluster_labels = clusterer.fit_predict(features)
                print(f"   DBSCAN clustering complete!")
            except (MemoryError, OSError) as e:
                print(f"   Memory/system resource error: {e}")
                print(f"   Reducing dataset size and retrying...")
                if num_points > 200000:
                    reduce_to = 200000
                    print(f"   Reducing to {reduce_to:,} points for clustering...")
                    sample_indices = np.random.choice(num_points, reduce_to, replace=False)
                    features = features[sample_indices]
                    original_indices = original_indices[sample_indices]
                    x_coords = x_coords[sample_indices]
                    y_coords = y_coords[sample_indices]
                    z_coords = z_coords[sample_indices]
                    num_points = reduce_to
                    
                    clusterer = DBSCAN(eps=eps_normalized, min_samples=min_samples, metric='euclidean', n_jobs=1)
                    cluster_labels = clusterer.fit_predict(features)
                    print(f"   DBSCAN clustering complete with reduced dataset!")
                else:
                    print(f"   Error: Dataset too large. Try reducing --max-points further.")
                    raise
            
        elif method.lower() == 'kmeans':
            # K-means: Fast, fixed number of clusters
            print(f"   Using K-means with {n_clusters} clusters (fast, fixed clusters)...")
            try:
                clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, n_jobs=1)
                cluster_labels = clusterer.fit_predict(features)
                print(f"   K-means clustering complete!")
            except Exception as e:
                print(f"   K-means error: {e}, trying MiniBatchKMeans...")
                clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
                cluster_labels = clusterer.fit_predict(features)
                print(f"   MiniBatchKMeans clustering complete!")
            
        elif method.lower() == 'minibatch':
            # MiniBatch K-means: Even faster for very large datasets
            print(f"   Using MiniBatchKMeans with {n_clusters} clusters (very fast for large datasets)...")
            batch_size = min(1000, num_points // 10)
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, n_init=3)
            cluster_labels = clusterer.fit_predict(features)
            print(f"   MiniBatchKMeans clustering complete!")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dbscan', 'kmeans', or 'minibatch'")
    else:
        # Simple distance-based clustering (fallback)
        print("   Using simple distance-based clustering (install scikit-learn for better results)...")
        eps_normalized = eps_spatial / max(x_range, y_range, z_range)
        cluster_labels = simple_clustering(features, eps_normalized, min_samples)
    
    # Get unique cluster labels (excluding noise points labeled as -1 for DBSCAN)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove noise (DBSCAN only)
    
    num_clusters = len(unique_clusters)
    noise_points = np.sum(cluster_labels == -1) if method.lower() == 'dbscan' else 0
    
    print(f"   Found {num_clusters} clusters")
    if noise_points > 0:
        print(f"   Noise points (not in any cluster): {noise_points:,}")
    
    # --- POCZĄTEK MODYFIKACJI ---
    # Set up output directory
    
    # Wymuszamy folder 'Clusters'
    output_dir = "Clusters"
    
    # Upewniamy się, że folder istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Ustawienie nazwy bazowej pliku wyjściowego
    base_name = os.path.splitext(os.path.basename(las_file_path))[0]
    
    # --- KONIEC MODYFIKACJI ---
    
    output_files = []
    
    # Export each cluster
    print(f"\nExporting clusters...")
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = np.sum(cluster_mask)
        
        # Skip very small clusters (unless using K-means which creates all clusters)
        if method.lower() != 'kmeans' and method.lower() != 'minibatch' and cluster_points < min_samples:
            continue
        
        # Get original indices for this cluster
        cluster_original_indices = original_indices[cluster_mask]
        
        # Create output filename
        output_filename = f"{base_name}_cluster_{cluster_id:04d}_{cluster_points}pts.las"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create new LAS file for this cluster
        out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        out_las.header.offsets = las.header.offsets
        out_las.header.scales = las.header.scales
        
        # Copy points in this cluster
        out_las.points = las.points[cluster_original_indices]
        
        # Update header with new bounds
        cluster_x = x_coords[cluster_mask]
        cluster_y = y_coords[cluster_mask]
        cluster_z = z_coords[cluster_mask]
        
        out_las.header.x_min = float(np.min(cluster_x))
        out_las.header.x_max = float(np.max(cluster_x))
        out_las.header.y_min = float(np.min(cluster_y))
        out_las.header.y_max = float(np.max(cluster_y))
        out_las.header.z_min = float(np.min(cluster_z))
        out_las.header.z_max = float(np.max(cluster_z))
        
        # Write to file
        out_las.write(output_path)
        output_files.append(output_path)
        
        # Calculate cluster statistics
        # Używamy np.ptp (peak-to-peak) zamiast max - min, dla prostoty
        range_sum = (np.ptp(cluster_x) + np.ptp(cluster_y) + np.ptp(cluster_z) + 0.001)
        cluster_density = cluster_points / range_sum if range_sum > 0 else 0
        
        print(f"   Cluster {cluster_id}: {cluster_points:,} points "
              f"(density: {cluster_density:.2f} pts/unit) -> {output_filename}")
    
    print(f"\nSuccessfully created {len(output_files)} cluster files!")
    print(f"   Output directory: {os.path.abspath(output_dir)}")
    
    return output_files


def simple_clustering(features, eps, min_samples):
    """
    Simple distance-based clustering (fallback when scikit-learn is not available).
    """
    n_points = len(features)
    cluster_labels = np.full(n_points, -1)  # -1 means unassigned
    cluster_id = 0
    
    for i in range(n_points):
        if cluster_labels[i] != -1:
            continue
        
        # Find neighbors
        distances = np.sqrt(np.sum((features - features[i]) ** 2, axis=1))
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) >= min_samples:
            # Start new cluster
            cluster_labels[neighbors] = cluster_id
            cluster_id += 1
    
    return cluster_labels


def main():
    """Main function to run the script."""
    if len(sys.argv) < 2:
        print("Usage: python las_color_cluster.py <path_to_las_file> [OPTIONS]")
        print("\nOptions:")
        print("   --n-clusters VALUE      Number of clusters for K-means (default: 50)")
        print("   --method METHOD       Clustering method: 'dbscan' (density, default), 'kmeans', or 'minibatch'")
        print("   --eps-spatial VALUE     Max spatial distance for DBSCAN density clustering (default: 1.0)")
        print("   --min-samples VALUE     Minimum points per cluster for DBSCAN (default: 10)")
        print("   --output-dir DIR        Output directory (default: same as input)")
        print("   --max-points VALUE      Max points to process (default: 10000000)")
        print("\nExamples:")
        print("   # Density-based clustering with DBSCAN (recommended for object extraction)")
        print("   python las_color_cluster.py data.las")
        print("   # Tighter density clustering (smaller objects)")
        print("   python las_color_cluster.py data.las --eps-spatial 0.5")
        print("   # Looser density clustering (larger objects)")
        print("   python las_color_cluster.py data.las --eps-spatial 2.0")
        print("   # K-means with fixed number of clusters")
        print("   python las_color_cluster.py data.las --method kmeans --n-clusters 50")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    
    # Parse arguments
    n_clusters = 50   # Default: good for object extraction
    method = 'dbscan'   # Default: density-based clustering for objects
    eps_spatial = 2.5   # Default: 1.0 units for density clustering
    min_samples = 200
    output_dir = None
    max_points = 200000
    
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
            output_dir=output_dir, # Ta wartość (None) jest ignorowana, bo została nadpisana wewnątrz funkcji
            max_points=max_points
        )
        print(f"\nAll done! Created {len(output_files)} cluster files.")
        return output_files
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()