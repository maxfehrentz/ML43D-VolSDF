import numpy as np
from plyfile import PlyData, PlyElement

def check_close_to_zero(array, threshold=1e-6):
    for vector in array:
        if abs(vector[3]) < threshold:
            print(vector)

loaded_shape = np.load("data/SdfSamples/VolSDF_Overfit/02691156/c58e9f5593a26f75e607cb6cb5eb37ba.npz")

positions = loaded_shape["neg"][:,:3]
sdf_values = loaded_shape["neg"][:,3:]

min_val = np.min(sdf_values)
max_val = np.max(sdf_values)

print(len(loaded_shape["neg"]) + len(loaded_shape["pos"]))

print(np.min(positions))
print(np.max(positions))
print(min_val, max_val)
check_close_to_zero(loaded_shape["neg"])
    
# Normalize to 0-1 range
color = (sdf_values - min_val) / (max_val - min_val)
    
    # Scale to 0-255 range and convert to uint8
uint8_color = (color * 255).astype(np.uint8)

vertex = np.array([(v[0], v[1], v[2], c[0], c[0], c[0]) 
                   for v, c in zip(positions, uint8_color)],
                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

# Create PlyElement
el = PlyElement.describe(vertex, 'vertex')

# Create PlyData
ply_data = PlyData([el])

# Write to file
ply_data.write('output_with_color.ply')