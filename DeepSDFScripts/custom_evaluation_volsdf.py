import numpy as np
import trimesh
from tqdm import tqdm
from simpleicp import PointCloud, SimpleICP

from deep_sdf.metrics import chamfer
from scipy.stats import wasserstein_distance_nd as wd
from scipy.spatial import cKDTree as KDTree

ground_truths = [
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/02691156/10155655850468db78d106ce0a280f87/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/02828884/1042d723dfc31ce5ec56aed2da084563/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/02958343/1005ca47e516495512da0dbf3c68e847/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/03001627/303a25778d48a0f671a782a4379556c7/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/03211117/1063cfe209bdaeb340ff33d80c1d7d1e/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/03636649/101d0e7dbd07d8247dfd6bf7196ba84d/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/03691459/101354f9d8dede686f7b08d9de913afe/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/04256520/1037fd31d12178d396f164a988ef37cc/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/04379243/1011e1c9812b84d2a9ed7bb5b55809f8/pointcloud.npz",
    "/home/jonas/DeepSDF/custom_eval_data/annotated_data/04530566/10212c1a94915e146fc883a34ed13b89/pointcloud.npz"
]

results = [
    "/home/jonas/data/volsdf_results/02691156/10155655850468db78d106ce0a280f87/surface_1000.obj",
    "/home/jonas/data/volsdf_results/02828884/1042d723dfc31ce5ec56aed2da084563/surface_1000.obj",
    "/home/jonas/data/volsdf_results/02958343/1005ca47e516495512da0dbf3c68e847/surface_1000.obj",
    "/home/jonas/data/volsdf_results/03001627/303a25778d48a0f671a782a4379556c7/surface_1000.obj",
    "/home/jonas/data/volsdf_results/03211117/1063cfe209bdaeb340ff33d80c1d7d1e/surface_1000.obj",
    "/home/jonas/data/volsdf_results/03636649/101d0e7dbd07d8247dfd6bf7196ba84d/surface_1000.obj",
    "/home/jonas/data/volsdf_results/03691459/101354f9d8dede686f7b08d9de913afe/surface_1000.obj",
    "/home/jonas/data/volsdf_results/04256520/1037fd31d12178d396f164a988ef37cc/surface_1000.obj",
    "/home/jonas/data/volsdf_results/04379243/1011e1c9812b84d2a9ed7bb5b55809f8/surface_1000.obj",
    "/home/jonas/data/volsdf_results/04530566/10212c1a94915e146fc883a34ed13b89/surface_1000.obj"
]

def load_pointcloud(filepath: str):
    points = np.load(filepath)["points"]
    points -= np.mean(points)
    diameter = np.max(points) - np.min(points)
    scale_factor = 2/diameter
    points *= scale_factor
    point_cloud =  trimesh.PointCloud(points)

    # Access the points as a numpy array
    return point_cloud   

def load_meshes_as_pointclouds(filepath: str, num_samples = 30000):
    # Load the mesh
    mesh = trimesh.load_mesh(filepath)  # Replace with your mesh file path

    # Sample points from the mesh surface
    points = mesh.sample(num_samples)
    
    # Zero center points
    points -= np.mean(points)
    diameter = np.max(points) - np.min(points)
    scale_factor = 2/diameter
    points *= scale_factor

    # Now 'points' is a numpy array of shape (num_points, 3) representing your point cloud

    # If you want to visualize the point cloud
    return trimesh.points.PointCloud(points)

def align_shapes(gt, res, sample_count=30000):
    sampled_gt_rows = np.random.choice(gt.vertices.shape[0], size=sample_count, replace=False)
    sampled_res_rows = np.random.choice(res.vertices.shape[0], size=sample_count, replace=False)
    pc_fix = PointCloud(gt.vertices[sampled_gt_rows], columns=["x", "y", "z"])
    pc_mov = PointCloud(res.vertices[sampled_res_rows], columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm!
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
    
    points = np.asarray(res.vertices)
    homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply transformation
    transformed_points = np.dot(homogeneous_points, H.T)
    # Normalize
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3:]
    
    return trimesh.points.PointCloud(transformed_points)
    

def compute_trimesh_chamfer(gt_points, gen_pts, offset = 0, scale  = 0, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    """

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    gen_points_sampled = gen_pts.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

def calculate_emd(gt_cloud, res_cloud, samples = 500):
    sampled_gt_rows = np.random.choice(gt_cloud.shape[0], size=samples, replace=False)
    subsampled_gt_points = gt_cloud[sampled_gt_rows]
    
    sampled_res_rows = np.random.choice(res_cloud.shape[0], size=samples, replace=False)
    subsampled_res_points = gt_cloud[sampled_res_rows]
    
    return wd(subsampled_gt_points, subsampled_res_points)

if __name__ == "__main__":
    
    chamfer_distances = []
    emd = []
    
    for index in tqdm(range(len(ground_truths))):
        gt = load_pointcloud(ground_truths[index])
        res = load_meshes_as_pointclouds(results[index])
        
        aligned_res = align_shapes(gt, res)
        
        chamfer_distances.append(compute_trimesh_chamfer(gt, aligned_res))
        emd.append(calculate_emd(gt, aligned_res))
        
    chamfer_distances = np.asarray(chamfer_distances)
    emd = np.asarray(emd)

    print("Mean(cd, emd):", np.mean(chamfer_distances), np.mean(emd))
    print("Median(cd, emd):", np.median(chamfer_distances), np.median(emd))