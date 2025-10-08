# test.py
"""
Test / inference script for CycleGAN.
- loads latest (or specified) checkpoint
- runs generator on testA (A->B) and testB (B->A)
- saves outputs to results/<name>/<which_epoch>/<phase>/
- displays a small grid inline (useful in Colab)
"""
import os
import argparse
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from models import ResnetGenerator

def get_transform_for_test(size=256):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def denorm(tensor):
    return (tensor + 1.0) / 2.0

def find_checkpoint(checkpoints_dir, name, which_epoch):
    model_dir = os.path.join(checkpoints_dir, name)
    if which_epoch == 'latest':
        ckpts = glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth'))
        if not ckpts:
            return None
        ckpt = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))[-1]
        return ckpt
    else:
        path = os.path.join(model_dir, which_epoch)
        return path if os.path.exists(path) else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./Dataset/horse2zebra')
    parser.add_argument('--name', type=str, default='horse2zebra_cyclegan')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--which_epoch', type=str, default='latest')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--show_samples', action='store_true', help='display sample grids inline (Colab)')
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = find_checkpoint(args.checkpoints_dir, args.name, args.which_epoch)
    if ckpt is None:
        raise FileNotFoundError("Checkpoint not found. Run train.py first or provide checkpoint path.")
    print("Loading checkpoint:", ckpt)
    state = torch.load(ckpt, map_location=device)

    # Create generators and load weights
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    netG_A2B.load_state_dict(state['netG_A2B'])
    netG_B2A.load_state_dict(state['netG_B2A'])
    netG_A2B.eval(); netG_B2A.eval()

    transform = get_transform_for_test(size=args.crop_size)

    # Process A -> B (testA)
    srcA = os.path.join(args.dataroot, 'testA')
    outA = os.path.join('results', args.name, os.path.basename(ckpt).replace('.pth',''), 'testA')
    os.makedirs(outA, exist_ok=True)
    filesA = sorted([f for f in os.listdir(srcA) if f.lower().endswith(('png','jpg','jpeg'))])

    sample_outputs = []  # for inline display
    with torch.no_grad():
        for file in filesA:
            img = Image.open(os.path.join(srcA, file)).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            fake = netG_A2B(x)
            save_image(denorm(fake[0]).clamp(0,1), os.path.join(outA, file))
            if len(sample_outputs) < 6:
                sample_outputs.append(torch.cat([x.cpu()[0], fake.cpu()[0]], dim=2))  # side-by-side

    # Process B -> A (testB)
    srcB = os.path.join(args.dataroot, 'testB')
    outB = os.path.join('results', args.name, os.path.basename(ckpt).replace('.pth',''), 'testB')
    os.makedirs(outB, exist_ok=True)
    filesB = sorted([f for f in os.listdir(srcB) if f.lower().endswith(('png','jpg','jpeg'))])

    with torch.no_grad():
        for file in filesB:
            img = Image.open(os.path.join(srcB, file)).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            fake = netG_B2A(x)
            save_image(denorm(fake[0]).clamp(0,1), os.path.join(outB, file))
            if len(sample_outputs) < 12:
                sample_outputs.append(torch.cat([x.cpu()[0], fake.cpu()[0]], dim=2))

    print("Saved results to:", outA, "and", outB)

    # Inline display of a small grid (useful in Colab)
    if args.show_samples and len(sample_outputs) > 0:
        grid = make_grid([(t + 1)/2 for t in sample_outputs], nrow=3)
        plt.figure(figsize=(12,8))
        plt.axis('off')
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.show()
