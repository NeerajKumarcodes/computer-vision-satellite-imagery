import torch

from losses import DiceLoss, BCEDiceLoss


logits = torch.randn(
    4,
    1,
    512,
    512,
)

targets = torch.randint(
    0,
    2,
    (4, 1, 512, 512),
).float()


dice_loss = DiceLoss()
combined_loss = BCEDiceLoss()


dice_value = dice_loss(logits, targets)
combined_value = combined_loss(logits, targets)


print("Dice Loss:", dice_value.item())
print("BCE + Dice Loss:", combined_value.item())