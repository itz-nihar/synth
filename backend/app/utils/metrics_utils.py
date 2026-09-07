import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
from PIL import Image
from typing import List, Tuple, Dict
from backend.app.config import DEVICE

# Lightweight feature extractor based on ResNet18
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC classification layer to get 512-dim feature embeddings
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)
            return feat.squeeze(-1).squeeze(-1)

_FEATURE_EXTRACTOR = None

def get_feature_extractor():
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = FeatureExtractor().to(DEVICE)
    return _FEATURE_EXTRACTOR

def extract_features_from_pil(images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
    if not images:
        return np.zeros((0, 512))
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    extractor = get_feature_extractor()
    all_features = []
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        tensors = torch.stack([transform(img.convert('RGB')) for img in batch_imgs]).to(DEVICE)
        feats = extractor(tensors).cpu().numpy()
        all_features.append(feats)
        
    return np.concatenate(all_features, axis=0)

def compute_fid(real_features: np.ndarray, synthetic_features: np.ndarray) -> float:
    if len(real_features) < 2 or len(synthetic_features) < 2:
        return 999.0  # Fallback penalty if dataset too small
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = synthetic_features.mean(axis=0), np.cov(synthetic_features, rowvar=False)
    
    diff = mu1 - mu2
    
    # Calculate matrix square root
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    # Numerical error fix for complex numbers
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real
            
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.clip(fid, 0, 1000.0))

def compute_ssim_psnr_metrics(real_imgs: List[Image.Image], synthetic_imgs: List[Image.Image]) -> Tuple[float, float]:
    if not real_imgs or not synthetic_imgs:
        return 0.0, 0.0
    
    # Convert PIL images to 2D float arrays for simple SSIM/PSNR calculation
    n_samples = min(len(real_imgs), len(synthetic_imgs), 20)
    ssim_list = []
    psnr_list = []
    
    for i in range(n_samples):
        r_arr = np.array(real_imgs[i].resize((64, 64)).convert('RGB')).astype(np.float32)
        s_arr = np.array(synthetic_imgs[i].resize((64, 64)).convert('RGB')).astype(np.float32)
        
        # PSNR
        mse = np.mean((r_arr - s_arr) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        psnr_list.append(psnr)
        
        # Simplified SSIM per channel
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        mu_r, mu_s = np.mean(r_arr), np.mean(s_arr)
        var_r, var_s = np.var(r_arr), np.var(s_arr)
        cov_rs = np.mean((r_arr - mu_r) * (s_arr - mu_s))
        
        ssim = ((2 * mu_r * mu_s + c1) * (2 * cov_rs + c2)) / ((mu_r**2 + mu_s**2 + c1) * (var_r + var_s + c2))
        ssim_list.append(np.clip(ssim, -1.0, 1.0))
        
    avg_ssim = float(np.mean(ssim_list))
    avg_psnr = float(np.mean(psnr_list))
    return avg_ssim, avg_psnr

def compute_histogram_similarity(real_imgs: List[Image.Image], synthetic_imgs: List[Image.Image]) -> float:
    if not real_imgs or not synthetic_imgs:
        return 0.0
    
    def get_hist(imgs):
        hist = np.zeros(256 * 3)
        for img in imgs[:30]:
            arr = np.array(img.convert('RGB'))
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c], bins=256, range=(0, 256))
                hist[c*256:(c+1)*256] += h
        hist_sum = hist.sum()
        return hist / hist_sum if hist_sum > 0 else hist
    
    h_real = get_hist(real_imgs)
    h_syn = get_hist(synthetic_imgs)
    
    # Intersection metric
    intersection = np.minimum(h_real, h_syn).sum()
    return float(np.clip(intersection, 0.0, 1.0))

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def compute_tstr_utility(real_imgs: List[Image.Image], synthetic_imgs: List[Image.Image]) -> Dict[str, float]:
    """
    TSTR (Train on Synthetic, Test on Real) utility evaluation.
    Create pseudo classification task (e.g. brightness high/low or spatial features),
    train classifier on synthetic dataset, and test performance on real test dataset.
    """
    if len(synthetic_imgs) < 4 or len(real_imgs) < 4:
        return {"tstr_accuracy": 0.5, "trtr_accuracy": 0.5, "utility_ratio": 1.0}
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def prepare_dataset(imgs):
        X, Y = [], []
        for img in imgs:
            t = transform(img.convert('RGB'))
            # Generate pseudo label based on mean pixel luminance
            label = 1 if t.mean().item() > 0.0 else 0
            X.append(t)
            Y.append(label)
        return torch.stack(X), torch.tensor(Y, dtype=torch.long)
    
    syn_X, syn_Y = prepare_dataset(synthetic_imgs)
    real_X, real_Y = prepare_dataset(real_imgs)
    
    # Train on Synthetic
    model_syn = SimpleClassifier().to(DEVICE)
    opt = torch.optim.Adam(model_syn.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    model_syn.train()
    dataset = torch.utils.data.TensorDataset(syn_X, syn_Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(16, len(syn_X)), shuffle=True)
    for _ in range(5):  # 5 fast epochs
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model_syn(bx), by).backward()
            opt.step()
            
    # Evaluate on Real
    model_syn.eval()
    with torch.no_grad():
        preds = model_syn(real_X.to(DEVICE)).argmax(dim=1).cpu()
        tstr_acc = float((preds == real_Y).float().mean().item())
        
    # Baseline TRTR (Train on Real)
    model_real = SimpleClassifier().to(DEVICE)
    opt_r = torch.optim.Adam(model_real.parameters(), lr=0.01)
    loader_r = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(real_X, real_Y), batch_size=min(16, len(real_X)), shuffle=True)
    for _ in range(5):
        for bx, by in loader_r:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_r.zero_grad()
            crit(model_real(bx), by).backward()
            opt_r.step()
    with torch.no_grad():
        preds_r = model_real(real_X.to(DEVICE)).argmax(dim=1).cpu()
        trtr_acc = float((preds_r == real_Y).float().mean().item())
        
    utility_ratio = float(np.clip(tstr_acc / max(trtr_acc, 0.01), 0.0, 1.2))
    return {
        "tstr_accuracy": round(tstr_acc, 4),
        "trtr_accuracy": round(trtr_acc, 4),
        "utility_ratio": round(utility_ratio, 4)
    }

def compute_nndr_privacy(real_features: np.ndarray, synthetic_features: np.ndarray) -> Dict[str, float]:
    """
    Nearest Neighbor Distance Ratio (NNDR) & Copying/Memorization score.
    Higher NNDR / distance means lower risk of memorization/copying.
    """
    if len(real_features) == 0 or len(synthetic_features) == 0:
        return {"nndr_score": 0.5, "copy_leakage_percent": 0.0, "privacy_score": 50.0}
    
    # Compute pairwise Euclidean distances between synthetic and real
    # Shape: (num_syn, num_real)
    dists = np.linalg.norm(synthetic_features[:, None, :] - real_features[None, :, :], axis=-1)
    
    # For each synthetic image, find 1st and 2nd nearest real neighbors
    sorted_dists = np.sort(dists, axis=1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1] if sorted_dists.shape[1] > 1 else d1 + 1e-5
    
    nndr_ratios = d1 / (d2 + 1e-5)
    mean_nndr = float(np.mean(nndr_ratios))
    
    # Detect exact or near-identical copies (distance below threshold)
    threshold = 0.05 * np.max(dists)
    copies = np.sum(d1 < threshold)
    leakage_pct = float((copies / len(synthetic_features)) * 100.0)
    
    # Privacy score out of 100
    # High NNDR and low leakage = high privacy protection
    privacy_score = float(np.clip(80.0 * mean_nndr + (100.0 - leakage_pct) * 0.2, 0.0, 100.0))
    
    return {
        "nndr_score": round(mean_nndr, 4),
        "copy_leakage_percent": round(leakage_pct, 2),
        "privacy_score": round(privacy_score, 2)
    }
