import torch

from unet import UNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet().to(device)

x = torch.randn(
    2,
    3,
    512,
    512,
    device=device,
)

with torch.no_grad():
    output = model(x)

print("Device:", device)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Model parameters:", sum(p.numel() for p in model.parameters()))