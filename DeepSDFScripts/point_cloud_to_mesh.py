import open3d as o3d
import trimesh
import numpy as np

points = np.load("/home/jonas/DeepSDF/custom_eval_data/annotated_data/02691156/10155655850468db78d106ce0a280f87/pointcloud.npz")["points"]

pcd = o3d.geometry.PointCloud()

# Set the points from the numpy array
pcd.points = o3d.utility.Vector3dVector(points)

pcd.estimate_normals()

# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist   

mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

tri_mesh.export('output_mesh.obj')