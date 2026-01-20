import tifffile
import numpy as np
import open3d as o3d
# Paths
tiff_path = "/home/zgp/Documents/m3dmpre/datasets/mvtec3d/foam/test/contamination/xyz/015.tiff"
ply_path = "foamcontamination015.ply"
# Load (x, y, z) TIFF
xyz = tifffile.imread(tiff_path)  # shape: (H, W, 3)
print(f"Loaded TIFF with shape: {xyz.shape}")
# Reshape to N×3 point list
points = xyz.reshape(-1, 3)
# Remove NaNs or invalid points
mask = np.isfinite(points).all(axis=1)
points = points[mask]
# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
# Save as PLY
o3d.io.write_point_cloud(ply_path, pcd)
print(f"Saved {ply_path} with {len(points)} points")
