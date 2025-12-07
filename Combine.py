"""
Optimized script to merge multiple LAS files while preserving original point positions.
"""

import laspy
import numpy as np
import os
import sys

INPUT_DIR = 'Clusters'
OUTPUT_FILENAME = 'Merged_Points.las'

# Get list of files to merge
las_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.las')]
las_files = [f for f in las_files if f != OUTPUT_FILENAME]

if not las_files:
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' not found.")
    else:
        print(f"Error: No .las files found in {os.path.abspath(INPUT_DIR)}.")
    sys.exit(1)

print(f"Found {len(las_files)} files to merge")

# Step 1: Read all files and find best point format
las_list = []
max_point_format = None
max_format_id = -1
max_header = None

for file_name in las_files:
    path = os.path.join(INPUT_DIR, file_name)
    try:
        las = laspy.read(path)
        las_list.append(las)
        
        if las.header.point_format.id > max_format_id:
            max_format_id = las.header.point_format.id
            max_point_format = las.header.point_format
            max_header = las.header
            
    except Exception as e:
        print(f"Error reading {file_name}: {e}. Skipping.")
        continue

if not las_list:
    print("Error: Could not read any .las files.")
    sys.exit(1)

print(f"Using Point Format ID {max_format_id} for merged file.")

# Step 2: Calculate combined bounds from ALL files to set correct offsets/scales
print("Calculating combined bounds from all files...")
all_x_min = []
all_x_max = []
all_y_min = []
all_y_max = []
all_z_min = []
all_z_max = []

for las in las_list:
    # Get real coordinates (not scaled integers)
    real_x = np.array(las.x)
    real_y = np.array(las.y)
    real_z = np.array(las.z)
    
    all_x_min.append(real_x.min())
    all_x_max.append(real_x.max())
    all_y_min.append(real_y.min())
    all_y_max.append(real_y.max())
    all_z_min.append(real_z.min())
    all_z_max.append(real_z.max())

# Combined bounds
combined_x_min = min(all_x_min)
combined_x_max = max(all_x_max)
combined_y_min = min(all_y_min)
combined_y_max = max(all_y_max)
combined_z_min = min(all_z_min)
combined_z_max = max(all_z_max)

print(f"Combined bounds: X=[{combined_x_min:.2f}, {combined_x_max:.2f}], "
      f"Y=[{combined_y_min:.2f}, {combined_y_max:.2f}], "
      f"Z=[{combined_z_min:.2f}, {combined_z_max:.2f}]")

# Step 3: Create merged file with appropriate offsets and scales
merged = laspy.create(point_format=max_point_format, file_version=max_header.version)

# Set offsets to the minimum values (standard LAS practice)
merged.header.offsets = [combined_x_min, combined_y_min, combined_z_min]

# Calculate appropriate scales based on the coordinate ranges
# Scale should be small enough to preserve precision but large enough to avoid overflow
# LAS uses int32 for coordinates, so max range is about 2.1 billion * scale
ranges = [
    combined_x_max - combined_x_min,
    combined_y_max - combined_y_min,
    combined_z_max - combined_z_min
]

# Use finest scale from files, but ensure it can handle the full range
best_scales = list(max_header.scales)
for las in las_list:
    for i in range(3):
        if las.header.scales[i] < best_scales[i]:
            best_scales[i] = las.header.scales[i]

# Verify scales can handle the range (int32 max is ~2.1e9)
# If range/scale > 2e9, we need a larger scale
for i in range(3):
    if ranges[i] > 0:
        max_coord_value = ranges[i] / best_scales[i]
        if max_coord_value > 2e9:
            # Scale up to fit in int32
            best_scales[i] = ranges[i] / 2e9

merged.header.scales = best_scales
print(f"Using offsets: {merged.header.offsets}")
print(f"Using scales: {merged.header.scales}")
print(f"Coordinate ranges: X={ranges[0]:.2f}, Y={ranges[1]:.2f}, Z={ranges[2]:.2f}")

# Step 4: Collect all point data and coordinates
print("Collecting points from all files...")
all_coords = []  # Store real coordinates
all_point_data = []  # Store point records

for i, las in enumerate(las_list):
    file_name = las_files[i] if i < len(las_files) else f"file_{i}"
    print(f"  Processing {i+1}/{len(las_list)}: {file_name} ({len(las.points):,} points)")
    
    # Get real coordinates (these preserve original positions)
    real_x = np.array(las.x, dtype=np.float64)
    real_y = np.array(las.y, dtype=np.float64)
    real_z = np.array(las.z, dtype=np.float64)
    
    all_coords.append((real_x, real_y, real_z))
    all_point_data.append(las)

# Step 5: Combine all coordinates and create merged points
print("Combining all points...")
total_points = sum(len(coords[0]) for coords in all_coords)
print(f"Total points to merge: {total_points:,}")

# Find a file with the same point format to use as template
template_las = None
for las in las_list:
    if las.header.point_format.id == max_point_format.id:
        template_las = las
        break

# Create points array by copying from template and expanding
if template_las is not None:
    # Start with template points and expand
    merged_points_array = np.zeros(total_points, dtype=template_las.points.array.dtype)
else:
    # Fallback: create from dtype
    point_dtype = max_point_format.dtype()
    merged_points_array = np.zeros(total_points, dtype=point_dtype)

current_idx = 0

for i, (las, (real_x, real_y, real_z)) in enumerate(zip(all_point_data, all_coords)):
    num_points = len(real_x)
    
    # Convert coordinates to new system
    new_X = np.round((real_x - merged.header.offsets[0]) / merged.header.scales[0]).astype(np.int32)
    new_Y = np.round((real_y - merged.header.offsets[1]) / merged.header.scales[1]).astype(np.int32)
    new_Z = np.round((real_z - merged.header.offsets[2]) / merged.header.scales[2]).astype(np.int32)
    
    # Set coordinates first
    merged_points_array["X"][current_idx:current_idx + num_points] = new_X
    merged_points_array["Y"][current_idx:current_idx + num_points] = new_Y
    merged_points_array["Z"][current_idx:current_idx + num_points] = new_Z
    
    # Copy point data dimension by dimension (works for all formats)
    # Iterate over dimensions in the merged format to ensure they all exist
    for dim in merged.point_format.dimension_names:
        if dim in ["X", "Y", "Z"]:
            # Skip coordinates - already set
            continue
        try:
            # Check if dimension exists in source file
            if dim in las.point_format.dimension_names:
                # Get the dimension data as numpy array
                dim_data = np.array(las.points[dim])
                merged_points_array[dim][current_idx:current_idx + num_points] = dim_data
            # If dimension doesn't exist in source, it will remain at default value (0)
        except (KeyError, ValueError, AttributeError):
            # Skip dimensions that can't be copied
            pass
    
    current_idx += num_points

# Assign points to merged LAS file
# Use the underlying array property if available, otherwise try direct assignment
if template_las is not None:
    # Create points by copying the structure from template and modifying
    # Get the underlying numpy array structure
    template_array = template_las.points.array
    # Create new array with same dtype
    new_points_array = np.zeros(total_points, dtype=template_array.dtype)
    # Copy our merged data
    for field_name in merged_points_array.dtype.names:
        if field_name in new_points_array.dtype.names:
            new_points_array[field_name] = merged_points_array[field_name]
    # Create PointRecord using the same class as template
    merged.points = template_las.points.__class__(
        new_points_array,
        merged.point_format,
        merged.header.scales,
        merged.header.offsets
    )

else:
    # Fallback: try to create PointRecord directly
    try:
        from laspy.point import PointRecord
        merged.points = PointRecord(merged_points_array, merged.point_format)
    except:
        # Last resort: direct assignment
        merged.points = merged_points_array

# Step 6: Update header bounds
print("Updating header bounds...")
merged.update_header()

# Verify bounds are correct and coordinates preserved
print(f"Final bounds: X=[{merged.header.x_min:.2f}, {merged.header.x_max:.2f}], "
      f"Y=[{merged.header.y_min:.2f}, {merged.header.y_max:.2f}], "
      f"Z=[{merged.header.z_min:.2f}, {merged.header.z_max:.2f}]")

# Verify coordinate conversion by checking a few points
print("\nVerifying coordinate preservation (checking first 3 points):")
for i in range(min(3, len(merged.points))):
    # Reconstruct real coordinates from stored integers
    stored_x = merged.points["X"][i] * merged.header.scales[0] + merged.header.offsets[0]
    stored_y = merged.points["Y"][i] * merged.header.scales[1] + merged.header.offsets[1]
    stored_z = merged.points["Z"][i] * merged.header.scales[2] + merged.header.offsets[2]
    print(f"  Point {i}: stored=({stored_x:.3f}, {stored_y:.3f}, {stored_z:.3f})")

# Step 7: Write output
print(f"Writing merged file to {OUTPUT_FILENAME}...")
merged.write(OUTPUT_FILENAME)

print(f"\n[OK] Successfully merged {len(las_list)} files.")
print(f"Total points: {len(merged.points):,}")
print(f"Output saved to: {os.path.abspath(OUTPUT_FILENAME)}")
