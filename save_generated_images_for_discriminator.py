
import os
import numpy as np
from PIL import Image

def save_generated_images_for_discriminator(images, epoch, output_dir="/content/drive/MyDrive/Honors_Thesis/Output/ARDM_EVAL"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert JAX DeviceArray to numpy if needed
    if hasattr(images, "device_buffer"):
        images = np.array(images)

    for i, img in enumerate(images):
        # Scale from [0.0, 1.0] float to [0, 255] uint8
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_path = os.path.join(output_dir, f"epoch{epoch}_img{i:04}.png")
        Image.fromarray(img_uint8).save(img_path)

    print(f"[DEBUG] Saved {len(images)} images to {output_dir}")
