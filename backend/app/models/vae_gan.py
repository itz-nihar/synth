import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Callable, Optional
from pathlib import Path
from backend.app.models.base import BaseGenerativeModel
from backend.app.config import DEVICE

class SobelEdgeLoss(nn.Module):
    """Calculates high-frequency gradient edge loss to eliminate blur and force sharp anatomical edges"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        return torch.abs(pred_x - target_x).mean() + torch.abs(pred_y - target_y).mean()

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 64, nc: int = 3):
        super().__init__()
        ndf = 32
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), # 16 -> 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1), # 8 -> 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(ndf * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(ndf * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 64, nc: int = 3):
        super().__init__()
        self.image_size = image_size
        ngf = 32
        self.fc = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 32 * 8, 4, 4)
        out = self.deconv(h)
        if out.shape[-1] != self.image_size or out.shape[-2] != self.image_size:
            out = F.interpolate(out, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return out

class VAEDiscriminator(nn.Module):
    def __init__(self, image_size: int = 64, nc: int = 3):
        super().__init__()
        ndf = 32
        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ndf * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out, feat

class VAEGANModel(BaseGenerativeModel):
    """
    Variational Autoencoder combined with Generative Adversarial Network (VAE-GAN).
    Learns continuous latent representations while penalizing blur via adversarial discriminator feedback.
    """
    def __init__(self, image_size: int = 64, latent_dim: int = 128, is_monochrome: bool = True):
        super().__init__(image_size, latent_dim)
        self.is_monochrome = is_monochrome
        self.encoder = VAEEncoder(latent_dim, image_size).to(DEVICE)
        self.decoder = VAEDecoder(latent_dim, image_size).to(DEVICE)
        self.discriminator = VAEDiscriminator(image_size).to(DEVICE)
        self.edge_loss_fn = SobelEdgeLoss().to(DEVICE)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.0003,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> dict:
        opt_enc_dec = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr)

        bce_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        last_loss_g, last_loss_d = 0.0, 0.0

        for epoch in range(1, epochs + 1):
            epoch_loss_g, epoch_loss_d = 0.0, 0.0
            total_batches = len(dataloader)
            if total_batches == 0:
                break

            for i, data in enumerate(dataloader):
                real_images = data[0] if isinstance(data, (list, tuple)) else data
                real_images = real_images.to(DEVICE)
                b_size = real_images.size(0)

                # Forward VAE
                mu, logvar = self.encoder(real_images)
                z = self.reparameterize(mu, logvar)
                recon_images = self.decoder(z)

                if self.is_monochrome:
                    recon_images = recon_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                z_p = torch.randn(b_size, self.latent_dim, device=DEVICE)
                fake_images = self.decoder(z_p)

                if self.is_monochrome:
                    fake_images = fake_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                # Train Discriminator
                self.discriminator.zero_grad()
                out_real, feat_real = self.discriminator(real_images)
                out_recon, _ = self.discriminator(recon_images.detach())
                out_fake, _ = self.discriminator(fake_images.detach())

                lbl_real = torch.full((b_size, 1), 0.9, device=DEVICE)
                lbl_fake = torch.full((b_size, 1), 0.0, device=DEVICE)

                l_d = bce_loss(out_real, lbl_real) + bce_loss(out_recon, lbl_fake) + bce_loss(out_fake, lbl_fake)
                l_d.backward()
                opt_disc.step()

                # Train Encoder & Decoder
                opt_enc_dec.zero_grad()
                out_recon, feat_recon = self.discriminator(recon_images)
                out_fake, _ = self.discriminator(fake_images)

                # Sharpness & Structure Losses: KL + L1 Reconstruction + Sobel Edge Loss + Feature Matching + GAN
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_size
                pixel_recon_loss = l1_loss(recon_images, real_images)
                edge_loss = self.edge_loss_fn(recon_images, real_images)
                feat_loss = mse_loss(feat_recon, feat_real.detach())
                g_loss = bce_loss(out_recon, lbl_real) + bce_loss(out_fake, lbl_real)

                # Heavy adversarial & edge weights force sharp discriminator edges over mean blur
                l_total_g = (kl_loss * 0.0001) + (pixel_recon_loss * 0.1) + (edge_loss * 2.0) + (feat_loss * 2.0) + (g_loss * 5.0)
                l_total_g.backward()
                opt_enc_dec.step()

                epoch_loss_g += l_total_g.item()
                epoch_loss_d += l_d.item()

            last_loss_g = epoch_loss_g / total_batches
            last_loss_d = epoch_loss_d / total_batches

            if progress_callback:
                progress_callback(epoch, epochs, last_loss_g, last_loss_d)

        self.is_trained = True
        return {"loss_g": round(last_loss_g, 4), "loss_d": round(last_loss_d, 4)}

    def generate_samples(self, num_samples: int = 16) -> List[Image.Image]:
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=DEVICE)
            fake_tensors = self.decoder(z).cpu()
            if self.is_monochrome:
                fake_tensors = fake_tensors.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            fake_tensors = (fake_tensors + 1.0) / 2.0
            fake_tensors = torch.clamp(fake_tensors, 0.0, 1.0)

        pil_images = []
        to_pil = transforms.ToPILImage()
        for i in range(num_samples):
            pil_images.append(to_pil(fake_tensors[i]))

        return pil_images

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'image_size': self.image_size,
            'latent_dim': self.latent_dim,
            'is_monochrome': self.is_monochrome
        }, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.is_monochrome = checkpoint.get('is_monochrome', True)
        self.is_trained = True
