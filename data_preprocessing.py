import os
import numpy as np
import shutil
from tqdm import tqdm

import subprocess

def preprocess_cameras(src_folder, dst_folder, num_scans, img_width, img_height):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Get the list of object categories
    obj_categories = [obj for obj in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, obj))]

    index_counter = 0

    # Initialize the progress bar for object categories
    with tqdm(total=len(obj_categories), desc='Object Categories', position=0, leave=True) as pbar_obj:
        for obj_category in obj_categories:
            obj_category_path = os.path.join(src_folder, obj_category)

            # Get the list of instances for the current object category
            instances = [inst for inst in os.listdir(obj_category_path) if os.path.isdir(os.path.join(obj_category_path, inst))]

            total_instance_count = min(len(instances), num_scans)
            
            # Initialize the progress bar for instances
            with tqdm(total=total_instance_count, desc=f'Processing {obj_category}', position=0, leave=True) as pbar_inst:
                for instance_index in range(total_instance_count):
                    instance_path = os.path.join(obj_category_path, instances[instance_index])
                    
                    # Path to the cameras.npz file
                    cameras_npz_path = os.path.join(instance_path, 'cameras.npz')
                    
                    # Load the cameras.npz file
                    camera_data = np.load(cameras_npz_path)

                    # Determine unique indices from the keys
                    indices = set()
                    for key in camera_data.files:
                        idx = key.split('_')[-1]
                        indices.add(idx)

                    # Create a dictionary to store the new camera data
                    new_camera_data = {}
                    new_scale_data = {}
                    
                    for idx in indices:
                        world_mat_key = f'world_mat_{idx}'
                        camera_mat_key = f'camera_mat_{idx}'
                        scale_mat_key = f'scale_mat_{idx}' if f'scale_mat_{idx}' in camera_data.files else None
                        
                        # Store the matrices
                        if world_mat_key not in camera_data.files or camera_mat_key not in camera_data.files:
                            continue
                            
                        world_mat = camera_data[world_mat_key]
                        camera_mat = camera_data[camera_mat_key]
                        scale_mat = camera_data[scale_mat_key] if scale_mat_key else np.identity(4)
                        
                        camera_mat[0][0] *= img_width/2
                        camera_mat[1][1] *= img_height/2
                        camera_mat[0][2] += img_width/2
                        camera_mat[1][2] += img_height/2
                        #camera_mat[2][2] *= -1
                    
                        new_camera_data[f'world_mat_{idx}'] = camera_mat @ world_mat
                        new_scale_data[f'scale_mat_{idx}'] = scale_mat
                    
                    # Create the new instance directory
                    new_instance_path = os.path.join(dst_folder, f'scan{index_counter}')
                    os.makedirs(new_instance_path, exist_ok=True)

                    # Copy image files to the new directory
                    src_images_path = os.path.join(instance_path, 'image')
                    dst_images_path = os.path.join(new_instance_path, 'image')
                    shutil.copytree(src_images_path, dst_images_path)

                    # Save the new cameras.npz file
                    np.savez(os.path.join(new_instance_path, 'cameras.npz'), **new_camera_data, **new_scale_data)

                    index_counter = index_counter + 1
                    
                    # Update the instance progress bar
                    pbar_inst.update(1)
            
            # Update the object category progress bar
            pbar_obj.update(1)


def normalize_cameras(root_dir):
    # Traverse the directory structure
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'cameras.npz':
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, 'cameras.npz')
                
                # Normalize the cameras.npz file
                result = subprocess.run([
                    "python", "data/preprocess/normalize_cameras.py",
                    "--input_cameras_file", input_file,
                    "--output_cameras_file", output_file
                ], capture_output=True, text=True)
                
                
if __name__=="__main__": 
    # Define input and output paths
    input_folder = os.path.join('.','data','NMR_Dataset')
    output_folder = os.path.join('.','data', 'NMR_Dataset_preprocessed')

    # Run preprocessing
    preprocess_cameras(input_folder, output_folder, 100, 64, 64)