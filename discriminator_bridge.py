import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from train_discriminator import Discriminator

folder = sys.argv[1]
if not os.path.isdir(folder):
    raise ValueError(f"Provided path is not a directory: {folder}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disc = Discriminator().to(device)
disc.load_state_dict(torch.load("/content/drive/MyDrive/Honors_Thesis/discriminator.pth"))
disc.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

scores = []
for fname in os.listdir(folder):
    if not fname.endswith(".png"): continue
    img = Image.open(os.path.join(folder, fname)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = disc(tensor).item()
    scores.append(score)

if not scores:
    print("No PNG images found.")
else:
    print(sum(scores) / len(scores))  # Average score
