# test.py
import os
import argparse
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from models import ResnetGenerator
from torchvision.utils import save_image

def get_transform_for_test(size=256):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

def denorm(t):
    return (t + 1.0) / 2.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./Dataset/horse2zebra')
    parser.add_argument('--name', type=str, default='horse2zebra_cyclegan')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--phase', type=str, default='test', choices=['test'])
    parser.add_argument('--which_epoch', type=str, default='latest')  # or 'checkpoint_epoch_100.pth'
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = os.path.join(args.checkpoints_dir, args.name)
    # find checkpoint
    if args.which_epoch == 'latest':
        # pick the checkpoint with largest epoch in name
        ckpts = glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth'))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found.")
        ckpt = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))[-1]
    else:
        ckpt = os.path.join(model_dir, args.which_epoch)
    print("Using checkpoint:", ckpt)
    state = torch.load(ckpt, map_location=device)

    # load generators
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    netG_A2B.load_state_dict(state['netG_A2B'])
    netG_B2A.load_state_dict(state['netG_B2A'])
    netG_A2B.eval(); netG_B2A.eval()

    transform = get_transform_for_test()
    src_dir = os.path.join(args.dataroot, args.phase + 'A')  # testA or testB
    out_dir = os.path.join('./results', args.name, args.which_epoch, args.phase + 'A')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('png','jpg','jpeg'))])
    for file in files:
        img = Image.open(os.path.join(src_dir, file)).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake = netG_A2B(x)
        save_image(denorm(fake[0]), os.path.join(out_dir, file))
    print("Saved results to", out_dir)
