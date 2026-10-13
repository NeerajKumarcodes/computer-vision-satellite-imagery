import torch

from unet import UNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet().to(device)

model.train()

# Start with batch size 4
images = torch.randn(
    4,
    3,
    512,
    512,
    device=device,
)

masks = torch.randint(
    0,
    2,
    (4, 1, 512, 512),
    device=device,
).float()


criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)


optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, masks)

loss.backward()

optimizer.step()


allocated = torch.cuda.memory_allocated() / (1024 ** 3)
reserved = torch.cuda.memory_reserved() / (1024 ** 3)


print("Device:", device)
print("Output shape:", outputs.shape)
print("Loss:", loss.item())
print(f"GPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved: {reserved:.2f} GB")