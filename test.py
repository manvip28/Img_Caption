import torch

checkpoint = torch.load("checkpoints/checkpoint_epoch8.pth", map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())
print("Loaded successfully!")