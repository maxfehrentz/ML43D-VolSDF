import torch

# # Load the model
# model = torch.load('/home/aleks/ml3d/ML43D-VolSDF/models/nmr_multi/2024_07_17_00_41_31/checkpoints/LatentCodes/latest.pth')

# # Print model architecture
# print(model['latent_codes'].shape)


# Load the model
model = torch.load('/home/aleks/ml3d/ML43D-VolSDF/models/nmr_multi/2024_07_17_00_41_31/checkpoints/ModelParameters/latest.pth')

# Write the model's string representation to a file
with open('model_info.txt', 'w') as f:
    f.write(str(model['model_state_dict']['implicit_network.z']))


print("Model information has been saved to 'model_info.txt'")
print(model['model_state_dict']['implicit_network.z'].shape)


