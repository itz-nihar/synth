import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Callable, Optional
from pathlib import Path
from backend.app.models.base import BaseGenerativeModel
from backend.app.config import DEVICE

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, image_size: int = 64, nc: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Smooth upsampling with Upsample + Conv2d to eliminate checkerboard artifacts
        if image_size == 32:
            ngf = 32
            self.fc = nn.Linear(latent_dim, ngf * 4 * 4 * 4)
            self.main = nn.Sequential(
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 4 -> 8
                nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 8 -> 16
                nn.Conv2d(ngf * 2, ngf, 3, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 16 -> 32
                nn.Conv2d(ngf, nc, 3, padding=1, bias=False),
                nn.Tanh()
            )
        else: # 64x64 default
            ngf = 64
            self.fc = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
            self.main = nn.Sequential(
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 4 -> 8
                nn.Conv2d(ngf * 8, ngf * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 8 -> 16
                nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 16 -> 32
                nn.Conv2d(ngf * 2, ngf, 3, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), # 32 -> 64
                nn.Conv2d(ngf, nc, 3, padding=1, bias=False),
                nn.Tanh()
            )

    def forward(self, input_tensor):
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
        h = self.fc(input_tensor)
        if self.image_size == 32:
            h = h.view(-1, 32 * 4, 4, 4)
        else:
            h = h.view(-1, 64 * 8, 4, 4)
        return self.main(h)


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size: int = 64, nc: int = 3):
        super().__init__()
        ndf = 32 if image_size == 32 else 64
        if image_size == 32:
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(ndf * 4, 1),
                nn.Sigmoid()
            )
        else: # 64x64
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(ndf * 8, 1),
                nn.Sigmoid()
            )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class GANModel(BaseGenerativeModel):
    def __init__(self, image_size: int = 64, latent_dim: int = 100, is_monochrome: bool = True):
        super().__init__(image_size, latent_dim)
        self.is_monochrome = is_monochrome
        self.generator = DCGANGenerator(latent_dim, image_size).to(DEVICE)
        self.discriminator = DCGANDiscriminator(image_size).to(DEVICE)

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
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

                label_real = torch.full((b_size, 1), 0.9, device=DEVICE)  # label smoothing
                label_fake = torch.full((b_size, 1), 0.0, device=DEVICE)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()
                output_real = self.discriminator(real_images)
                loss_d_real = criterion(output_real, label_real)

                noise = torch.randn(b_size, self.latent_dim, device=DEVICE)
                fake_images = self.generator(noise)
                
                # Enforce monochrome if medical scan
                if self.is_monochrome:
                    fake_images = fake_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                output_fake = self.discriminator(fake_images.detach())
                loss_d_fake = criterion(output_fake, label_fake)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.generator.zero_grad()
                output = self.discriminator(fake_images)
                loss_g = criterion(output, label_real)
                loss_g.backward()
                optimizer_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

            last_loss_g = epoch_loss_g / total_batches
            last_loss_d = epoch_loss_d / total_batches

            if progress_callback:
                progress_callback(epoch, epochs, last_loss_g, last_loss_d)

        self.is_trained = True
        return {"loss_g": round(last_loss_g, 4), "loss_d": round(last_loss_d, 4)}

    def generate_samples(self, num_samples: int = 16) -> List[Image.Image]:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=DEVICE)
            fake_tensors = self.generator(noise).cpu()
            
            if self.is_monochrome:
                fake_tensors = fake_tensors.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            # Unnormalize from [-1, 1] to [0, 1]
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
