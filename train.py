# train.py
"""
Optimized CycleGAN training script (PyTorch, AMP)
Assumes models.py (ResnetGenerator, NLayerDiscriminator, init_weights),
dataset.py (UnpairedImageDataset, get_transform) and utils.py (ImagePool, save_sample)
are in the same directory.
"""
import os
import argparse
import itertools
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler

from models import ResnetGenerator, NLayerDiscriminator, init_weights
from dataset import UnpairedImageDataset, get_transform
from utils import ImagePool, save_sample

# -----------------------
# Utility helpers
# -----------------------
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is None:
            continue
        for p in net.parameters():
            p.requires_grad = requires_grad

def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

def find_latest_checkpoint(checkpoints_dir, name):
    model_dir = os.path.join(checkpoints_dir, name)
    if not os.path.isdir(model_dir):
        return None
    ckpts = glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth'))
    if not ckpts:
        return None
    ckpt = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))[-1]
    return ckpt

# -----------------------
# Training main
# -----------------------
def main(args):
    # Basic device & cuDNN tuning
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    torch.backends.cudnn.benchmark = True

    # Mixed precision scaler
    scaler = GradScaler()

    # Transforms and dataset
    transform = get_transform(load_size=args.load_size, crop_size=args.crop_size)
    dataset = UnpairedImageDataset(args.dataroot, phase='train', transform=transform)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        prefetch_factor=args.prefetch_factor,
                        drop_last=True)

    # Models
    netG_A2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    netG_B2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    netD_A = NLayerDiscriminator(input_nc=3, ndf=args.ndf, n_layers=args.n_layers_D).to(device)
    netD_B = NLayerDiscriminator(input_nc=3, ndf=args.ndf, n_layers=args.n_layers_D).to(device)

    init_weights(netG_A2B, init_type=args.init_type, init_gain=args.init_gain)
    init_weights(netG_B2A, init_type=args.init_type, init_gain=args.init_gain)
    init_weights(netD_A, init_type=args.init_type, init_gain=args.init_gain)
    init_weights(netD_B, init_type=args.init_type, init_gain=args.init_gain)

    # Losses
    criterion_GAN = nn.MSELoss().to(device)    # LSGAN
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                             lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Schedulers
    scheduler_G = get_scheduler(optimizer_G, args.n_epochs, args.n_epochs_decay)
    scheduler_D_A = get_scheduler(optimizer_D_A, args.n_epochs, args.n_epochs_decay)
    scheduler_D_B = get_scheduler(optimizer_D_B, args.n_epochs, args.n_epochs_decay)

    # Image pools
    fake_A_pool = ImagePool(pool_size=args.pool_size)
    fake_B_pool = ImagePool(pool_size=args.pool_size)

    # Labels (for LSGAN)
    def real_label_tensor(x):
        return torch.ones_like(x, device=device)
    def fake_label_tensor(x):
        return torch.zeros_like(x, device=device)

    # Resume if requested
    start_epoch = 1
    if args.continue_train:
        ckpt = find_latest_checkpoint(args.checkpoints_dir, args.name)
        if ckpt:
            print("Loading checkpoint:", ckpt)
            state = torch.load(ckpt, map_location=device)
            netG_A2B.load_state_dict(state['netG_A2B'])
            netG_B2A.load_state_dict(state['netG_B2A'])
            netD_A.load_state_dict(state['netD_A'])
            netD_B.load_state_dict(state['netD_B'])
            optimizer_G.load_state_dict(state['optimizer_G'])
            optimizer_D_A.load_state_dict(state['optimizer_D_A'])
            optimizer_D_B.load_state_dict(state['optimizer_D_B'])
            start_epoch = state.get('epoch', 1) + 1
        else:
            print("No checkpoint found, starting from scratch.")

    total_epochs = args.n_epochs + args.n_epochs_decay
    accum_steps = max(1, args.accum_steps)

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        netG_A2B.train(); netG_B2A.train(); netD_A.train(); netD_B.train()
        loop = tqdm(enumerate(loader, 1), total=len(loader), desc=f"Epoch {epoch}/{total_epochs}")

        for i, data in loop:
            real_A = data['A'].to(device, non_blocking=True)
            real_B = data['B'].to(device, non_blocking=True)

            # ------------------
            #  Train Generators (accumulation supported)
            # ------------------
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad(set_to_none=True)

            with autocast():
                # Identity
                idt_A = netG_B2A(real_A)
                idt_B = netG_A2B(real_B)
                loss_idt = (criterion_identity(idt_A, real_A) + criterion_identity(idt_B, real_B)) * args.lambda_identity

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake_B = netD_B(fake_B)
                loss_G_A2B = criterion_GAN(pred_fake_B, real_label_tensor(pred_fake_B))

                fake_A = netG_B2A(real_B)
                pred_fake_A = netD_A(fake_A)
                loss_G_B2A = criterion_GAN(pred_fake_A, real_label_tensor(pred_fake_A))

                # Cycle
                rec_A = netG_B2A(fake_B)
                rec_B = netG_A2B(fake_A)
                loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * args.lambda_cycle

                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle + loss_idt

            # scale for gradient accumulation
            loss_G_scaled = loss_G / accum_steps
            scaler.scale(loss_G_scaled).backward()

            if (i % accum_steps) == 0:
                scaler.step(optimizer_G)
                scaler.update()
                optimizer_G.zero_grad(set_to_none=True)

            # -----------------------
            #  Train Discriminators
            # -----------------------
            # D_A
            set_requires_grad(netD_A, True)
            optimizer_D_A.zero_grad(set_to_none=True)
            with autocast():
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, real_label_tensor(pred_real))

                fake_A_for_pool = fake_A_pool.query(fake_A)
                pred_fake = netD_A(fake_A_for_pool.detach())
                loss_D_fake = criterion_GAN(pred_fake, fake_label_tensor(pred_fake))

                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
            scaler.update()

            # D_B
            set_requires_grad(netD_B, True)
            optimizer_D_B.zero_grad(set_to_none=True)
            with autocast():
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, real_label_tensor(pred_real))

                fake_B_for_pool = fake_B_pool.query(fake_B)
                pred_fake = netD_B(fake_B_for_pool.detach())
                loss_D_fake = criterion_GAN(pred_fake, fake_label_tensor(pred_fake))

                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
            scaler.update()

            # -----------------------
            #  Logging & samples
            # -----------------------
            loop.set_postfix({
                'loss_G': f"{loss_G.item():.4f}",
                'loss_D_A': f"{loss_D_A.item():.4f}",
                'loss_D_B': f"{loss_D_B.item():.4f}"
            })

            # Save sample images periodically (use CPU tensors)
            if (i % args.sample_interval) == 0:
                sample_dir = os.path.join(args.checkpoints_dir, args.name, 'samples')
                os.makedirs(sample_dir, exist_ok=True)
                sample_path = os.path.join(sample_dir, f'epoch{epoch:03d}_iter{i:04d}.jpg')
                # save_sample expects cpu tensors in [-1,1]
                save_sample(real_A.detach().cpu(), fake_B.detach().cpu(),
                            rec_A.detach().cpu(), real_B.detach().cpu(),
                            fake_A.detach().cpu(), rec_B.detach().cpu(),
                            sample_path)

        # update LR schedulers each epoch
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Save checkpoint
        if epoch % args.save_epoch_freq == 0 or epoch == total_epochs:
            save_dir = os.path.join(args.checkpoints_dir, args.name)
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'netG_A2B': netG_A2B.state_dict(),
                'netG_B2A': netG_B2A.state_dict(),
                'netD_A': netD_A.state_dict(),
                'netD_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training finished.")

# -----------------------
# Argparse
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./Dataset/horse2zebra', help='path to dataset root')
    parser.add_argument('--name', type=str, default='horse2zebra_cyclegan', help='experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='where to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size (try 1/2/4)')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--n_epochs', type=int, default=100, help='initial epochs with fixed LR')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='epochs to linearly decay LR')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--ngf', type=int, default=64, help='generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters')
    parser.add_argument('--n_blocks', type=int, default=9, help='resnet blocks in generator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='PatchGAN layers')
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--pool_size', type=int, default=50)
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--load_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--gpu', action='store_true', help='use GPU if available')
    args = parser.parse_args()
    main(args)
