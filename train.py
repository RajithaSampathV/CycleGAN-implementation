# train.py
import os
import argparse
import itertools
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import ResnetGenerator, NLayerDiscriminator, init_weights
from dataset import UnpairedImageDataset, get_transform
from utils import ImagePool, save_sample

def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad

def main(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    # dataset + dataloader
    transform = get_transform(args.load_size, args.crop_size)
    dataset = UnpairedImageDataset(args.dataroot, phase='train', transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # models
    netG_A2B = ResnetGenerator().to(device)
    netG_B2A = ResnetGenerator().to(device)
    netD_A = NLayerDiscriminator().to(device)
    netD_B = NLayerDiscriminator().to(device)
    init_weights(netG_A2B); init_weights(netG_B2A); init_weights(netD_A); init_weights(netD_B)

    # losses
    criterion_GAN = nn.MSELoss().to(device)  # LSGAN (least-squares)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    # optimizers
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # schedulers
    scheduler_G = get_scheduler(optimizer_G, args.n_epochs, args.n_epochs_decay)
    scheduler_D_A = get_scheduler(optimizer_D_A, args.n_epochs, args.n_epochs_decay)
    scheduler_D_B = get_scheduler(optimizer_D_B, args.n_epochs, args.n_epochs_decay)

    # pools
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    # training loop params
    total_epochs = args.n_epochs + args.n_epochs_decay
    lambda_cycle = args.lambda_cycle
    lambda_id = args.lambda_identity

    start_epoch = 1
    global_step = 0
    for epoch in range(start_epoch, total_epochs + 1):
        netG_A2B.train(); netG_B2A.train(); netD_A.train(); netD_B.train()
        loop = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
        for i, data in enumerate(loop):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            batch_size = real_A.size(0)

            # ------------------
            #  Train Generators
            # ------------------
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()

            # Identity loss
            idt_A = netG_B2A(real_A)
            idt_B = netG_A2B(real_B)
            loss_idt = (criterion_identity(idt_A, real_A) + criterion_identity(idt_B, real_B)) * lambda_id

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_G_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_G_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle loss
            rec_A = netG_B2A(fake_B)
            rec_B = netG_A2B(fake_A)
            loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * lambda_cycle

            # total generators loss
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle + loss_idt
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            set_requires_grad(netD_A, True)
            optimizer_D_A.zero_grad()
            # real
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            # fake (from pool)
            fake_A_for_pool = fake_A_pool.query(fake_A)
            pred_fake = netD_A(fake_A_for_pool.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            set_requires_grad(netD_B, True)
            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            fake_B_for_pool = fake_B_pool.query(fake_B)
            pred_fake = netD_B(fake_B_for_pool.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # logging
            loop.set_postfix({
                'loss_G': f"{loss_G.item():.3f}",
                'loss_D_A': f"{loss_D_A.item():.3f}",
                'loss_D_B': f"{loss_D_B.item():.3f}"
            })

            # save sample every N steps
            if (i + 1) % args.sample_interval == 0:
                sample_dir = os.path.join(args.checkpoints_dir, args.name, 'samples')
                os.makedirs(sample_dir, exist_ok=True)
                sample_path = os.path.join(sample_dir, f'epoch{epoch:03d}_iter{i+1:04d}.jpg')
                save_sample(real_A.cpu(), fake_B.cpu(), rec_A.cpu(), real_B.cpu(), fake_A.cpu(), rec_B.cpu(), sample_path)

            global_step += 1

        # update schedulers
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # save checkpoints each epoch (or every few epochs)
        if epoch % args.save_epoch_freq == 0 or epoch == total_epochs:
            save_dir = os.path.join(args.checkpoints_dir, args.name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'netG_A2B': netG_A2B.state_dict(),
                'netG_B2A': netG_B2A.state_dict(),
                'netD_A': netD_A.state_dict(),
                'netD_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict()
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"Saved checkpoint to {save_dir}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./Dataset/horse2zebra', help='path to dataset root')
    parser.add_argument('--name', type=str, default='horse2zebra_cyclegan', help='experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='where to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=100, help='initial epochs with lr')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='epochs to decay lr')
    parser.add_argument('--load_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)  # often set 0.5 * lambda_cycle (here 5)
    parser.add_argument('--sample_interval', type=int, default=200, help='save sample every N iterations')
    parser.add_argument('--save_epoch_freq', type=int, default=5)
    parser.add_argument('--gpu', action='store_true', help='use GPU if available')
    args = parser.parse_args()
    main(args)
