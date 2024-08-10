import numpy as np
import trimesh

# from deep_sdf.metrics import chamfer
from scipy.stats import wasserstein_distance_nd as wd
# 
from scipy.spatial import cKDTree as KDTree

ground_truths_3_scenes = [
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/train3/scan0/pointcloud.npz", #scan0
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/train3/scan1/pointcloud.npz", #scan1
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/train3/scan2/pointcloud.npz" #scan2
]

ground_truths_val_10 = [
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33c9e81a88866451f4fb6842b3610149/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33c6568fd4de5aaf1e623da3c4e40c05/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33d955301966e4215ebedace13b486c4/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33ddbeb13fecf7868405970680284869/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33e11a7dec64dd42d91ca512389cc0a0/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33faf711ed54a4d3db22b838c125a50b/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/33fff5b8d113cca41b950a59358e57bd/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/34a89777594d3a61b2440702f5566974/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/34c656eeca31045724a182d01c698394/pointcloud.npz",
    "/home/aleks/ml3d/ML43D-VolSDF/data/NMR_planes/val10/34c669182c8c9a2623fc69eefd95e6d3/pointcloud.npz"
]


results_3_scenes_volsdf = [
    "/home/aleks/ml3d/ML43D-VolSDF/exps/exps_plane/nmr_0/2024_07_20_23_59_48/plots/surface_1000.ply", # scan0
    "/home/aleks/ml3d/ML43D-VolSDF/exps/exps_plane/nmr_1/2024_07_21_01_52_22/plots/surface_1000.ply", # scan1
    "/home/aleks/ml3d/ML43D-VolSDF/exps/exps_plane/nmr_2/2024_07_21_08_55_05/plots/surface_1000.ply" # scan2
]

results_3_scenes = [
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_3_infer/2024_07_22_06_49_00/inference_plots/surface_inference_24.ply", # scan0
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_3_infer/2024_07_22_06_49_00/inference_plots/surface_inference_48.ply", # scan1
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_3_infer/2024_07_22_06_49_00/inference_plots/surface_inference_0.ply" # scan2

]


results_val_10_colmap  =   [

]

results_val_10 = [
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_48.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_168.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_96.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_14_48_59/inference_plots/surface_inference_76.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_0.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_144.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_192.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_24.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_13_58_47/inference_plots/surface_inference_120.ply",
    "/home/aleks/ml3d/ML43D-VolSDF/exps/nmr_multi_infer/2024_07_22_14_56_53/inference_plots/surface_inference_220.ply"
]


def load_pointcloud(filepath: str):
    point_file = np.load(filepath)
    # for k in point_file.files:
    #     print(k)
    point_cloud = point_file['points']

    # point_cloud = trimesh.load(filepath)

    # Access the points as a numpy array
    return trimesh.points.PointCloud(point_cloud)

def load_meshes_as_pointclouds(filepath: str, num_samples = 100000):
    # Load the mesh
    mesh = trimesh.load_mesh(filepath)  # Replace with your mesh file path

    # Sample points from the mesh surface
    points = mesh.sample(num_samples)

    # Now 'points' is a numpy array of shape (num_points, 3) representing your point cloud

    # If you want to visualize the point cloud
    return trimesh.points.PointCloud(points)
    

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

    file_path = '/home/aleks/ml3d/ML43D-VolSDF/code/eval.txt'
    
    # chamfer_distances = []
    # emd = []
    
    # for index in range(len(ground_truths_3_scenes)):
    #     gt = load_pointcloud(ground_truths_3_scenes[index])
    #     res = load_meshes_as_pointclouds(results_3_scenes_volsdf[index])
        
    #     chamfer_distances.append(compute_trimesh_chamfer(gt, res))
    #     emd.append(calculate_emd(gt, res))
        
    # chamfer_distances = np.asarray(chamfer_distances)
    # emd = np.asarray(emd)


    # print("Mean(cd, emd):", np.mean(chamfer_distances), np.mean(emd))
    # print("Median(cd, emd):", np.median(chamfer_distances), np.median(emd))
    # with open(file_path, 'a') as file:
    #     # For mean values
    #     mean_cd = np.mean(chamfer_distances)
    #     mean_emd = np.mean(emd)
    #     formatted_string = f"VolSDF Mean(cd, emd): {mean_cd:.6f}, {mean_emd:.6f}\n"
    #     file.write(formatted_string)

    #     # For median values
    #     median_cd = np.median(chamfer_distances)
    #     median_emd = np.median(emd)
    #     formatted_string = f"VolSDF Median(cd, emd): {median_cd:.6f}, {median_emd:.6f}\n"
    #     file.write(formatted_string)


    # chamfer_distances = []
    # emd = []
    
    # for index in range(len(ground_truths_3_scenes)):
    #     gt = load_pointcloud(ground_truths_3_scenes[index])
    #     res = load_meshes_as_pointclouds(results_3_scenes[index])
        
    #     chamfer_distances.append(compute_trimesh_chamfer(gt, res))
    #     emd.append(calculate_emd(gt, res))
        
    # chamfer_distances = np.asarray(chamfer_distances)
    # emd = np.asarray(emd)

    # print("Mean(cd, emd):", np.mean(chamfer_distances), np.mean(emd))
    # print("Median(cd, emd):", np.median(chamfer_distances), np.median(emd))

    # with open(file_path, 'a') as file:
    #     # For mean values
    #     mean_cd = np.mean(chamfer_distances)
    #     mean_emd = np.mean(emd)
    #     formatted_string = f"Ours Mean(cd, emd): {mean_cd:.6f}, {mean_emd:.6f}\n"
    #     file.write(formatted_string)

    #     # For median values
    #     median_cd = np.median(chamfer_distances)
    #     median_emd = np.median(emd)
    #     formatted_string = f"Ours Median(cd, emd): {median_cd:.6f}, {median_emd:.6f}\n"
    #     file.write(formatted_string)


    # chamfer_distances = []
    # emd = []

    

    # for index in range(len(ground_truths_val_10)):
    #     gt = load_pointcloud(ground_truths_val_10[index])
    #     res = load_meshes_as_pointclouds(results_val_10_colmap[index])
        
    #     chamfer_distances.append(compute_trimesh_chamfer(gt, res))
    #     emd.append(calculate_emd(gt, res))
    
    # chamfer_distances = np.asarray(chamfer_distances)
    # emd = np.asarray(emd)

    # print("Mean(cd, emd):", np.mean(chamfer_distances), np.mean(emd))
    # print("Median(cd, emd):", np.median(chamfer_distances), np.median(emd))

    # with open(file_path, 'a') as file:
    #     # For mean values
    #     mean_cd = np.mean(chamfer_distances)
    #     mean_emd = np.mean(emd)
    #     formatted_string = f"Ours val 10 Mean(cd, emd): {mean_cd:.6f}, {mean_emd:.6f}\n"
    #     file.write(formatted_string)

    #     # For median values
    #     median_cd = np.median(chamfer_distances)
    #     median_emd = np.median(emd)
    #     formatted_string = f"Ours val 10 Median(cd, emd): {median_cd:.6f}, {median_emd:.6f}\n"
    #     file.write(formatted_string)
          

    chamfer_distances = []
    emd = []

    for index in range(len(ground_truths_val_10)):
        gt = load_pointcloud(ground_truths_val_10[index])
        res = load_meshes_as_pointclouds(results_val_10[index])
        
        chamfer_distances.append(compute_trimesh_chamfer(gt, res))
        emd.append(calculate_emd(gt, res))
    
    chamfer_distances = np.asarray(chamfer_distances)
    emd = np.asarray(emd)

    print("Mean(cd, emd):", np.mean(chamfer_distances), np.mean(emd))
    print("Median(cd, emd):", np.median(chamfer_distances), np.median(emd))

    with open(file_path, 'a') as file:
        # For mean values
        mean_cd = np.mean(chamfer_distances)
        mean_emd = np.mean(emd)
        formatted_string = f"Ours val 10 Mean(cd, emd): {mean_cd:.6f}, {mean_emd:.6f}\n"
        file.write(formatted_string)

        # For median values
        median_cd = np.median(chamfer_distances)
        median_emd = np.median(emd)
        formatted_string = f"Ours val 10 Median(cd, emd): {median_cd:.6f}, {median_emd:.6f}\n"
        file.write(formatted_string)

    