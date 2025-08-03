
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
import clip

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1), torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(), torch.nn.Linear(8 * 8 * 128, 1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def evaluate_generated_images(eval_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = Discriminator().to(device)
    disc.load_state_dict(torch.load("/content/drive/MyDrive/Honors_Thesis/discriminator.pth"))
    disc.eval()

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    images, scores = [], []
    for fname in sorted(os.listdir(eval_path)):
        if not fname.endswith(".png"): continue
        img = Image.open(os.path.join(eval_path, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        pred = disc(tensor)
        scores.append(pred.item())
        images.append(tensor.squeeze(0))

    avg_score = np.mean(scores)
    fake_imgs = torch.stack(images)
    real_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True),
        batch_size=len(fake_imgs), shuffle=True
    )
    real_imgs, _ = next(iter(real_loader))

    fid = FrechetInceptionDistance(feature=2048).to(device)
    iscore = InceptionScore().to(device)
    fid.update(real_imgs.to(device), real=True)
    fid.update(fake_imgs.to(device), real=False)
    iscore.update(fake_imgs.to(device))

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    sim_scores = []
    for img in fake_imgs:
        pil = transforms.ToPILImage()(img.cpu())
        clip_tensor = preprocess(pil).unsqueeze(0).to(device)
        image_feat = clip_model.encode_image(clip_tensor)
        text_feat = clip_model.encode_text(clip.tokenize(["a photo of an object"]).to(device))
        sim = torch.cosine_similarity(image_feat, text_feat).item()
        sim_scores.append(sim)

    result = {
        "avg_discriminator_score": float(avg_score),
        "fid": float(fid.compute().item()),
        "is": float(iscore.compute()[0].item()),
        "clip": float(np.mean(sim_scores))
    }
    np.save(os.path.join(eval_path, "metrics.npy"), result)
    print("[PyTorch] Saved metrics to metrics.npy")

if __name__ == "__main__":
    try:
        evaluate_generated_images("/content/drive/MyDrive/Honors_Thesis/Output/ARDM_EVAL")
    except Exception as e:
        import traceback
        print("Error in evaluate_generated_images:")
        traceback.print_exc()
