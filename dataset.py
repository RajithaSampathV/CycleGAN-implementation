# dataset.py
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_transform(load_size=286, crop_size=256):
    transform_list = [
        transforms.Resize(load_size, Image.BICUBIC),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)

class UnpairedImageDataset(Dataset):
    """
    Returns dict: {'A': tensor, 'B': tensor, 'A_paths': path, 'B_paths': path}
    Unpaired: B is randomly sampled for each A.
    """
    def __init__(self, root, phase='train', transform=None):
        super().__init__()
        self.dir_A = os.path.join(root, phase + 'A')
        self.dir_B = os.path.join(root, phase + 'B')
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A)
                               if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B)
                               if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        assert len(self.A_paths) > 0 and len(self.B_paths) > 0, "Empty dataset dirs"
        self.transform = transform

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, index):
        A_path = self.A_paths[index % len(self.A_paths)]
        B_path = random.choice(self.B_paths)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if self.transform:
            A = self.transform(A_img)
            B = self.transform(B_img)
        else:
            A = transforms.ToTensor()(A_img)
            B = transforms.ToTensor()(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
