# utils.py
import os
import random
import torch
from torchvision.utils import save_image

class ImagePool:
    """History of generated images for discriminator stability (pool_size default 50)."""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
        self.num_imgs = 0

    def query(self, images):
        """
        images: tensor (batch, C, H, W)
        returns: tensor (batch, C, H, W) sampled from pool or current images
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.images.append(img.clone())
                self.num_imgs += 1
                return_images.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = img.clone()
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return torch.cat(return_images, dim=0).to(images.device)

def save_sample(real_A, fake_B, rec_A, real_B, fake_A, rec_B, out_path, ncol=3):
    """
    Save a grid showing: realA | fakeB | recA  and realB | fakeA | recB
    """
    # Denormalize: inputs are in [-1,1]
    def denorm(x):
        return (x + 1.0) / 2.0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row1 = torch.cat([denorm(real_A), denorm(fake_B), denorm(rec_A)], dim=0)
    row2 = torch.cat([denorm(real_B), denorm(fake_A), denorm(rec_B)], dim=0)
    # stack rows vertically
    grid = torch.cat([row1, row2], dim=1)  # careful: concatenation shape depends on batch
    # If batch >1, just save first item per row
    # We'll create a simple image by selecting first item from each of the 6 images:
    imgs = [real_A[0], fake_B[0], rec_A[0], real_B[0], fake_A[0], rec_B[0]]
    # create grid and save
    save_image([ (img+1)/2 for img in imgs ], out_path, nrow=3)
