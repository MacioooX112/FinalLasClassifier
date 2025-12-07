"""
Script to generalize/downsample LiDAR point cloud to reduce the number of points.
Supports multiple downsampling methods.
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


def random_sample(las_file_path, target_points, output_file="general.las"):
    """
    Randomly sample points from the LAS file.
    
    Args:
        las_file_path (str): Path to the input .las file
        target_points (int): Target number of points
        output_file (str): Path to save the output .las file (optional)
    """
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    total_points = len(las.points)
    print(f"  Total points: {total_points:,}")
    
    if target_points >= total_points:
        print(f"  Target points ({target_points:,}) >= total points. No downsampling needed.")
        return las_file_path
    
    # Random sampling
    sample_indices = np.random.choice(total_points, target_points, replace=False)
    sample_indices = np.sort(sample_indices)  # Sort for better performance
    
    print(f"  Randomly sampling {target_points:,} points ({target_points/total_points*100:.2f}%)...")
    
    # Create output file
    if output_file is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_file = f"general.las"
    
    # Create new LAS file
    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header.offsets = las.header.offsets
    out_las.header.scales = las.header.scales
    
    # Copy sampled points
    out_las.points = las.points[sample_indices]
    
    # Update header bounds
    x_coords = np.array(las.x)[sample_indices]
    y_coords = np.array(las.y)[sample_indices]
    z_coords = np.array(las.z)[sample_indices]
    
    out_las.header.x_min = float(np.min(x_coords))
    out_las.header.x_max = float(np.max(x_coords))
    out_las.header.y_min = float(np.min(y_coords))
    out_las.header.y_max = float(np.max(y_coords))
    out_las.header.z_min = float(np.min(z_coords))
    out_las.header.z_max = float(np.max(z_coords))
    
    out_las.write(output_file)
    print(f"  ✓ Saved to: {output_file}")
    
    return output_file


def grid_downsample(las_file_path, grid_size, output_file=None):
    """
    Downsample using a regular grid - keep one point per grid cell.
    
    Args:
        las_file_path (str): Path to the input .las file
        grid_size (float): Size of the grid cell (same units as coordinates)
        output_file (str): Path to save the output .las file (optional)
    """
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)
    z_coords = np.array(las.z)
    
    total_points = len(x_coords)
    print(f"  Total points: {total_points:,}")
    print(f"  Grid size: {grid_size}")
    
    # Calculate grid indices
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Create grid indices
    x_grid = ((x_coords - x_min) / grid_size).astype(int)
    y_grid = ((y_coords - y_min) / grid_size).astype(int)
    
    # Create unique grid cell identifiers
    grid_cells = x_grid * 1000000 + y_grid  # Combine x and y into unique ID
    
    # Find first occurrence of each grid cell
    unique_cells, unique_indices = np.unique(grid_cells, return_index=True)
    
    sampled_points = len(unique_indices)
    print(f"  Grid downsampling: {sampled_points:,} points ({sampled_points/total_points*100:.2f}%)")
    
    # Create output file
    if output_file is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_file = f"{base_name}_generalized_grid_{grid_size}.las"
    
    # Create new LAS file
    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header.offsets = las.header.offsets
    out_las.header.scales = las.header.scales
    
    # Copy sampled points
    out_las.points = las.points[unique_indices]
    
    # Update header bounds
    out_las.header.x_min = float(np.min(x_coords[unique_indices]))
    out_las.header.x_max = float(np.max(x_coords[unique_indices]))
    out_las.header.y_min = float(np.min(y_coords[unique_indices]))
    out_las.header.y_max = float(np.max(y_coords[unique_indices]))
    out_las.header.z_min = float(np.min(z_coords[unique_indices]))
    out_las.header.z_max = float(np.max(z_coords[unique_indices]))
    
    out_las.write(output_file)
    print(f"  ✓ Saved to: {output_file}")
    
    return output_file


def percentage_sample(las_file_path, percentage, output_file=None):
    """
    Sample a percentage of points from the LAS file.
    
    Args:
        las_file_path (str): Path to the input .las file
        percentage (float): Percentage of points to keep (0-100)
        output_file (str): Path to save the output .las file (optional)
    """
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    total_points = len(las.points)
    target_points = int(total_points * (percentage / 100.0))
    
    print(f"  Total points: {total_points:,}")
    print(f"  Keeping {percentage}% = {target_points:,} points...")
    
    return random_sample(las_file_path, target_points, output_file)


def generalize(las_file_path, method='random', target_points=None, percentage=None, grid_size=None, output_file=None):
    """
    Generalize/downsample a LAS file using various methods.
    
    Args:
        las_file_path (str): Path to the input .las file
        method (str): Downsampling method ('random', 'percentage', 'grid')
        target_points (int): Target number of points (for 'random' method)
        percentage (float): Percentage to keep (for 'percentage' method, 0-100)
        grid_size (float): Grid cell size (for 'grid' method)
        output_file (str): Path to save the output .las file (optional)
    """
    if method == 'random':
        if target_points is None:
            raise ValueError("target_points must be specified for 'random' method")
        return random_sample(las_file_path, target_points, output_file)
    
    elif method == 'percentage':
        if percentage is None:
            raise ValueError("percentage must be specified for 'percentage' method")
        if not (0 < percentage <= 100):
            raise ValueError("percentage must be between 0 and 100")
        return percentage_sample(las_file_path, percentage, output_file)
    
    elif method == 'grid':
        if grid_size is None:
            raise ValueError("grid_size must be specified for 'grid' method")
        return grid_downsample(las_file_path, grid_size, output_file)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'random', 'percentage', or 'grid'")


def main():
    """Main function to run the script."""
    if len(sys.argv) < 3:
        print("Usage: python las_generalize.py <path_to_las_file> <method> [options]")
        print("\nMethods:")
        print("  random <target_points>     Randomly sample N points")
        print("  percentage <percent>       Keep X% of points (e.g., 10 for 10%)")
        print("  grid <grid_size>           Grid-based downsampling (one point per grid cell)")
        print("\nOptions:")
        print("  --output <file>            Output file path (optional)")
        print("\nExamples:")
        print("  # Randomly sample 1 million points")
        print("  python las_generalize.py data.las random 1000000")
        print("  # Keep 10% of points")
        print("  python las_generalize.py data.las percentage 10")
        print("  # Grid downsampling with 1.0 unit grid")
        print("  python las_generalize.py data.las grid 1.0")
        print("  # With custom output file")
        print("  python las_generalize.py data.las random 500000 --output small_data.las")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    method = sys.argv[2].lower()
    
    output_file = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    try:
        if method == 'random':
            if len(sys.argv) < 4:
                print("Error: target_points required for 'random' method")
                sys.exit(1)
            target_points = int(sys.argv[3])
            result = generalize(las_file_path, method='random', target_points=target_points, output_file=output_file)
        
        elif method == 'percentage':
            if len(sys.argv) < 4:
                print("Error: percentage required for 'percentage' method")
                sys.exit(1)
            percentage = float(sys.argv[3])
            result = generalize(las_file_path, method='percentage', percentage=percentage, output_file=output_file)
        
        elif method == 'grid':
            if len(sys.argv) < 4:
                print("Error: grid_size required for 'grid' method")
                sys.exit(1)
            grid_size = float(sys.argv[3])
            result = generalize(las_file_path, method='grid', grid_size=grid_size, output_file=output_file)
        
        else:
            print(f"Error: Unknown method '{method}'. Use 'random', 'percentage', or 'grid'")
            sys.exit(1)
        
        print(f"\n✓ Generalization complete! Output: {result}")
        return result
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

