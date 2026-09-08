import torch
import torch.nn as nn
import torch.optim as optim
import math
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Callable, Optional
from pathlib import Path
from backend.app.models.base import BaseGenerativeModel
from backend.app.config import DEVICE

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, time_emb_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.inc = nn.Conv2d(in_channels, 64, 3, padding=1)

        self.down1 = ResBlock(64, 64, time_emb_dim)
        self.down2_conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = ResBlock(128, 128, time_emb_dim)

        self.mid1 = ResBlock(128, 128, time_emb_dim)
        self.mid2 = ResBlock(128, 128, time_emb_dim)

        self.up2_conv = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up2 = ResBlock(128, 64, time_emb_dim)
        self.outc = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x, time):
        t_emb = self.time_mlp(time)

        h1 = self.inc(x)
        h1 = self.down1(h1, t_emb)

        h2 = self.down2_conv(h1)
        h2 = self.down2(h2, t_emb)

        h_mid = self.mid1(h2, t_emb)
        h_mid = self.mid2(h_mid, t_emb)

        h_up = self.up2_conv(h_mid)
        h_up = torch.cat([h_up, h1], dim=1)
        h_up = self.up2(h_up, t_emb)

        return self.outc(h_up)

class DiffusionModel(BaseGenerativeModel):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    Learns forward noise schedule and predicts noise via time-conditioned U-Net.
    """
    def __init__(self, image_size: int = 64, timesteps: int = 100, is_monochrome: bool = True):
        super().__init__(image_size=image_size, latent_dim=timesteps)
        self.timesteps = timesteps
        self.is_monochrome = is_monochrome
        self.unet = ImprovedUNet(in_channels=3, time_emb_dim=64).to(DEVICE)

        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.0002,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> dict:
        optimizer = optim.Adam(self.unet.parameters(), lr=lr)
        criterion = nn.MSELoss()
        last_loss = 0.0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            total_batches = len(dataloader)
            if total_batches == 0:
                break

            for i, data in enumerate(dataloader):
                x_start = data[0] if isinstance(data, (list, tuple)) else data
                x_start = x_start.to(DEVICE)
                b_size = x_start.size(0)

                t = torch.randint(0, self.timesteps, (b_size,), device=DEVICE).long()
                noise = torch.randn_like(x_start)

                x_noisy = self.q_sample(x_start, t, noise)
                predicted_noise = self.unet(x_noisy, t)

                loss = criterion(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            last_loss = epoch_loss / total_batches
            if progress_callback:
                progress_callback(epoch, epochs, last_loss, 0.0)

        self.is_trained = True
        return {"loss_g": round(last_loss, 4), "loss_d": 0.0}

    @torch.no_grad()
    def p_sample_step(self, x, t_index: int, t_prev_index: Optional[int] = None, eta: float = 0.0):
        batch_size = x.shape[0]
        t = torch.full((batch_size,), t_index, device=DEVICE, dtype=torch.long)

        predicted_noise = self.unet(x, t)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]

        # Estimate x_0 from x_t and predicted_noise
        x_0_pred = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / (sqrt_alpha_cumprod_t + 1e-8)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        if t_prev_index is None or t_index == 0:
            return x_0_pred

        alpha_cumprod_t = self.alphas_cumprod[t_index]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev_index] if t_prev_index >= 0 else torch.tensor(1.0, device=DEVICE)
        
        sigma_t = eta * torch.sqrt(torch.clamp((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t + 1e-8) * (1.0 - alpha_cumprod_t / (alpha_cumprod_prev + 1e-8)), min=0.0))
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - (sigma_t ** 2), min=0.0)) * predicted_noise
        
        noise = torch.randn_like(x) if eta > 0 else 0.0
        x_prev = torch.sqrt(alpha_cumprod_prev) * x_0_pred + dir_xt + sigma_t * noise
        return x_prev

    def generate_samples(self, num_samples: int = 16) -> List[Image.Image]:
        self.unet.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 3, self.image_size, self.image_size), device=DEVICE)
            
            # Fast DDIM strided timesteps (10 steps instead of 100) for instant response (< 1 sec)
            num_steps = 10
            step_size = max(1, self.timesteps // num_steps)
            strided_steps = list(range(0, self.timesteps, step_size))
            
            for i in reversed(range(len(strided_steps))):
                t_curr = strided_steps[i]
                t_prev = strided_steps[i - 1] if i > 0 else 0
                x = self.p_sample_step(x, t_curr, t_prev_index=t_prev, eta=0.0)

            if self.is_monochrome:
                x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            # Standard Tanh normalization mapping [-1.0, 1.0] -> [0.0, 1.0] to preserve true black background and tissue contrast
            x = (x + 1.0) / 2.0
            x = torch.clamp(x.cpu(), 0.0, 1.0)

        pil_images = []
        to_pil = transforms.ToPILImage()
        for i in range(num_samples):
            pil_images.append(to_pil(x[i]))

        return pil_images

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'unet': self.unet.state_dict(),
            'image_size': self.image_size,
            'timesteps': self.timesteps,
            'is_monochrome': self.is_monochrome
        }, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.unet.load_state_dict(checkpoint['unet'])
        self.is_monochrome = checkpoint.get('is_monochrome', True)
        self.is_trained = True


