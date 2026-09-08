import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Callable, Optional
from pathlib import Path
from backend.app.models.base import BaseGenerativeModel
from backend.app.config import DEVICE

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, image_size: int = 64, nc: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        ngf = 64

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.upsample = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
        h = self.fc(input_tensor).view(-1, 64 * 8, 4, 4)
        out = self.upsample(h)
        if out.shape[-1] != self.image_size or out.shape[-2] != self.image_size:
            out = torch.nn.functional.interpolate(out, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return out


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size: int = 64, nc: int = 3):
        super().__init__()
        ndf = 64
        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        feat = self.features(input_tensor)
        out = self.classifier(feat)
        return out.view(-1, 1)


class GANModel(BaseGenerativeModel):
    """
    Standard Deep Convolutional Generative Adversarial Network (DCGAN).
    Learns spatial structures and texture distributions from latent Gaussian noise.
    """
    def __init__(self, image_size: int = 64, latent_dim: int = 100, is_monochrome: bool = True):
        super().__init__(image_size, latent_dim)
        self.is_monochrome = is_monochrome
        self.generator = DCGANGenerator(latent_dim=latent_dim, image_size=image_size, nc=3).to(DEVICE)
        self.discriminator = DCGANDiscriminator(image_size=image_size, nc=3).to(DEVICE)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.0002,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> dict:
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

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

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()
                label_real = torch.full((b_size, 1), 0.9, device=DEVICE)  # Label smoothing
                output_real = self.discriminator(real_images)
                loss_d_real = criterion(output_real, label_real)

                noise = torch.randn(b_size, self.latent_dim, 1, 1, device=DEVICE)
                fake_images = self.generator(noise)

                label_fake = torch.full((b_size, 1), 0.0, device=DEVICE)
                output_fake = self.discriminator(fake_images.detach())
                loss_d_fake = criterion(output_fake, label_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.generator.zero_grad()
                label_gen = torch.full((b_size, 1), 0.9, device=DEVICE)
                output_gen = self.discriminator(fake_images)
                loss_g = criterion(output_gen, label_gen)
                loss_g.backward()
                optimizer_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

            last_loss_g = epoch_loss_g / total_batches
            last_loss_d = epoch_loss_d / total_batches

            if progress_callback:
                progress_callback(epoch, epochs, last_loss_g, last_loss_d)

        self.is_trained = True

        # Store real dataset latent representations mapped from discriminator features
        latent_list = []
        with torch.no_grad():
            for data in dataloader:
                imgs = data[0] if isinstance(data, (list, tuple)) else data
                imgs = imgs.to(DEVICE)
                feat = self.discriminator.features(imgs)
                feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).view(imgs.size(0), -1)
                # Trim/pad to latent_dim
                if feat_pooled.shape[1] >= self.latent_dim:
                    z_vec = feat_pooled[:, :self.latent_dim]
                else:
                    z_vec = torch.cat([feat_pooled, torch.zeros(imgs.size(0), self.latent_dim - feat_pooled.shape[1], device=DEVICE)], dim=1)
                latent_list.append(z_vec.view(imgs.size(0), self.latent_dim, 1, 1))
        self.latent_bank = torch.cat(latent_list, dim=0) if latent_list else None

        return {"loss_g": round(last_loss_g, 4), "loss_d": round(last_loss_d, 4)}

    def generate_samples(self, num_samples: int = 16) -> List[Image.Image]:
        self.generator.eval()
        with torch.no_grad():
            if hasattr(self, 'latent_bank') and self.latent_bank is not None and len(self.latent_bank) > 1:
                N = len(self.latent_bank)
                idx1 = torch.randint(0, N, (num_samples,))
                idx2 = torch.randint(0, N, (num_samples,))
                alpha = torch.rand((num_samples, 1, 1, 1), device=DEVICE)
                z1 = self.latent_bank[idx1]
                z2 = self.latent_bank[idx2]
                noise = alpha * z1 + (1.0 - alpha) * z2 + torch.randn_like(z1) * 0.10
            else:
                noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=DEVICE)

            fake_tensors = self.generator(noise).cpu()

            if self.is_monochrome:
                fake_tensors = fake_tensors.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            # Standard Tanh normalization mapping [-1.0, 1.0] -> [0.0, 1.0] to preserve true black background and tissue contrast
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
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'image_size': self.image_size,
            'latent_dim': self.latent_dim,
            'is_monochrome': self.is_monochrome
        }, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.is_monochrome = checkpoint.get('is_monochrome', True)
        self.is_trained = True
