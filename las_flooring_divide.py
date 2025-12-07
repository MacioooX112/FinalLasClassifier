"""
Script to find clusters in LiDAR point cloud based on color and spatial distance.
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





def Split_Selector(las_file_path, output_dir="Objects", Fraction_of_height=12):
    # If file path is relative or just filename, look in Objects directory
    if not os.path.isabs(las_file_path):
        # Check if file exists in Objects directory
        objects_path = os.path.join("Objects", las_file_path)
        if os.path.exists(objects_path):
            las_file_path = objects_path
        elif not os.path.exists(las_file_path):
            # If file doesn't exist in current dir, try Objects directory
            objects_path = os.path.join("Objects", os.path.basename(las_file_path))
            if os.path.exists(objects_path):
                las_file_path = objects_path
    
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"File not found: {las_file_path}")

    if not las_file_path.lower().endswith('.las'):
        raise ValueError(f"File must be a .las file: {las_file_path}")

    # Set up output directory - always use Objects directory
    if output_dir is None or output_dir == "":
        output_dir = "Objects"
    
    # Ensure Objects directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    z_coords = np.array(las.z)

    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    z_range = z_max - z_min

    threshold = z_min + z_range / Fraction_of_height
    mask = z_coords <= threshold
    inv_mask = ~mask


    las_class2 = laspy.LasData(las.header.copy())
    las_class2.points = las.points[mask].copy()
    las_class2.classification[:] = 2

    output_class2 = os.path.join(output_dir, "Class2_points.las")
    las_class2.write(output_class2)

    las_rest = laspy.LasData(las.header.copy())
    las_rest.points = las.points[inv_mask].copy()

    output_rest = os.path.join(output_dir, "Remaining_points.las")
    las_rest.write(output_rest)

    print(f"Saved {mask.sum()} class-2 points  {output_class2}")
    print(f"Saved {inv_mask.sum()} remaining points  {output_rest}")

    return [output_class2, output_rest]
def main():

    """Main function to run the script."""
    if len(sys.argv) < 2:
        print("Usage: python las_color_cluster.py <path_to_las_file> [OPTIONS]")
        print("\nOptions:")
        print("  --n-clusters VALUE      Number of clusters to create (default: 100)")
        print(
            "  --method METHOD        Clustering method: 'kmeans' (fast, default), 'minibatch' (very fast), or 'dbscan' (variable clusters)")
        print("  --eps-color VALUE      Max color distance for DBSCAN (default: 30)")
        print("  --min-samples VALUE    Minimum points per cluster for DBSCAN (default: 10)")
        print("  --output-dir DIR       Output directory (default: Objects)")
        print("  --max-points VALUE     Max points to process (default: 200000)")
        print("\nExamples:")
        print("  # Fast K-means clustering with 100 clusters (recommended for object extraction)")
        print("  python las_color_cluster.py data.las")
        print("  # Custom number of clusters")
        print("  python las_color_cluster.py data.las --n-clusters 50")
        print("  # Very fast MiniBatch K-means for huge datasets")
        print("  python las_color_cluster.py data.las --method minibatch --n-clusters 100")
        print("  # DBSCAN (variable number of clusters)")
        print("  python las_color_cluster.py data.las --method dbscan --eps-color 20")
        print("  Parametr funkcji wywolanej")
        #print(Tablica_nazw_funkcji)
        sys.exit(1)

    las_file_path = sys.argv[1]

    # Parse arguments
    n_clusters = 2  # Default: good for object extraction
    method = 'kmeans'  # Default: fastest and best for fixed number of clusters
    eps_color = 55
    min_samples = 10
    output_dir = "Objects"  # Default to Objects directory
    max_points = 10000000

    if '--n-clusters' in sys.argv:
        idx = sys.argv.index('--n-clusters')
        if idx + 1 < len(sys.argv):
            n_clusters = int(sys.argv[idx + 1])

    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1].lower()

    if '--eps-color' in sys.argv:
        idx = sys.argv.index('--eps-color')
        if idx + 1 < len(sys.argv):
            eps_color = float(sys.argv[idx + 1])

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

        output_files = Split_Selector(
                    las_file_path,
                    output_dir=output_dir,
                    Fraction_of_height=12)

        print(f"\nAll done! Created {len(output_files)} cluster files.")
        return output_files
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


