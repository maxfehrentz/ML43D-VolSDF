import torch
import argparse
import numpy as np
import json
import os
import trimesh
from skimage.measure import marching_cubes
from pathlib import Path

import deep_sdf.workspace as ws

def evaluate_model_on_grid(model, latent_code, device, grid_resolution, export_path):
    x_range = y_range = z_range = np.linspace(-1., 1., grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    stacked = torch.from_numpy(np.hstack((grid_x[:, np.newaxis], grid_y[:, np.newaxis], grid_z[:, np.newaxis]))).float().to(device)
    stacked_split = torch.split(stacked, 32 ** 3, dim=0)
    sdf_values = []
    for points in stacked_split:
        with torch.no_grad():
            sdf = model(torch.cat([latent_code.unsqueeze(0).expand(points.shape[0], -1), points], 1))
        sdf_values.append(sdf.detach().cpu())
    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape((grid_resolution, grid_resolution, grid_resolution))
    if 0 < sdf_values.min() or 0 > sdf_values.max():
        vertices, faces = [], []
    else:
        vertices, faces, _, _ = marching_cubes(sdf_values, level=0)
    if export_path is not None:
        Path(export_path).parent.mkdir(exist_ok=True)
        trimesh.Trimesh(vertices=vertices, faces=faces).export(export_path)
    return vertices, faces

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--scenes",
        "-s",
        dest="scene_count",
        default=10,
        help="The number of scenes the network was trained on",
    )
    
    args = arg_parser.parse_args()
    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    
    specs = json.load(open(specs_filename))
    
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    
    latent_size = specs["CodeLength"]
    
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
     
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(os.path.join(args.experiment_directory, "ModelParameters/latest.pth"))
    
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()
    
    device = torch.device(0)
    lat_vecs = torch.nn.Embedding(int(args.scene_count), latent_size)
    data = torch.load(os.path.join(args.experiment_directory, "LatentCodes/latest.pth"))
    lat_vecs.load_state_dict(data["latent_codes"])
    lat_vecs.to(device)
    latent_vectors_for_vis = lat_vecs(torch.LongTensor(range(lat_vecs.num_embeddings)).to(device))
    for latent_idx in range(latent_vectors_for_vis.shape[0]):
        # create mesh and save to disk
        evaluate_model_on_grid(decoder, latent_vectors_for_vis[latent_idx, :], device, 256, f'./sampledSDF{latent_idx}.obj')