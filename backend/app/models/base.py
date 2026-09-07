from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from PIL import Image
import torch
from pathlib import Path

class BaseGenerativeModel(ABC):
    def __init__(self, image_size: int = 64, latent_dim: int = 100):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.is_trained = False

    @abstractmethod
    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.0002,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> dict:
        """
        Train generative model.
        progress_callback signature: (current_epoch, total_epochs, loss_g, loss_d)
        """
        pass

    @abstractmethod
    def generate_samples(self, num_samples: int = 16) -> List[Image.Image]:
        """
        Generate num_samples synthetic PIL images.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path):
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path):
        pass
