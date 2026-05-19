# ============================================================
# SRMamba-T Satellite SR — TEST / INFERENCE SCRIPT
# ============================================================
# Run this AFTER training. It loads your saved checkpoint and
# evaluates on:
#   1. DIV2K validation set   (standard benchmark)
#   2. Satellite images       (HR_0.5m / LR_2m pairs)
#
# Outputs:
#   - PSNR / SSIM table for both datasets
#   - Visual comparison grids (LR | Bicubic | SR | HR)
#   - Per-image results CSV
# ============================================================

import os, glob, math, random, csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.system("pip install tifffile -q")
try:
    import tifffile; TIFF_OK = True
except: TIFF_OK = False

torch.manual_seed(42); random.seed(42); np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")


# ─────────────────────────────────────────
# PATHS  — exact structure from your Kaggle
# ─────────────────────────────────────────
CHECKPOINT  = "/kaggle/working/sat_sr_model.pth"   # your trained weights

# Satellite test images
SAT_HR      = "/kaggle/input/4x-satellite-image-super-resolution/HR_0.5m"
SAT_LR      = "/kaggle/input/4x-satellite-image-super-resolution/LR_2m"

# DIV2K validation (test split)
DIV_BASE    = "/kaggle/input/div2k-dataset-for-super-resolution/Dataset"
DIV_VAL_HR  = os.path.join(DIV_BASE, "DIV2K_valid_HR")
DIV_VAL_LR  = os.path.join(DIV_BASE, "DIV2K_valid_LR_bicubic_X4")

SCALE       = 4
IN_CH       = 3
NUM_FEAT    = 64
NUM_BLOCKS  = 8

OUT_DIR     = "/kaggle/working/test_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# MODEL  (must match training architecture)
# ─────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
    def forward(self, x): return x + self.body(x)

class ChannelAttention(nn.Module):
    def __init__(self, nf, r=16):
        super().__init__()
        mid = max(nf // r, 4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nf, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, nf), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x).view(x.shape[0], x.shape[1], 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], 1)))

class DeepFeatureBlock(nn.Module):
    def __init__(self, nf, nb):
        super().__init__()
        self.res  = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])
        self.ca   = ChannelAttention(nf)
        self.sa   = SpatialAttention()
        self.tail = nn.Conv2d(nf, nf, 3, 1, 1)
    def forward(self, x):
        f = self.res(x); f = self.ca(f); f = self.sa(f)
        return x + self.tail(f)

class SatelliteSRNet(nn.Module):
    def __init__(self, scale=4, nf=64, nb=8, in_ch=3):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.deep    = DeepFeatureBlock(nf, nb)
        self.fusion  = nn.Conv2d(nf, nf, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True),
            nn.Conv2d(nf, in_ch, 3, 1, 1)
        )
    def forward(self, x):
        s = self.shallow(x)
        d = self.deep(s)
        return self.up(self.fusion(d + s)).clamp(0, 1)

# ── Load checkpoint ──────────────────────────────────────────
model = SatelliteSRNet(scale=SCALE, nf=NUM_FEAT, nb=NUM_BLOCKS, in_ch=IN_CH).to(DEVICE)

if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print(f"✓ Loaded checkpoint: {CHECKPOINT}")
else:
    print(f"✗ Checkpoint not found at {CHECKPOINT}")
    print("  Make sure you have run training first!")
    raise FileNotFoundError(CHECKPOINT)

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")


# ─────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def list_images(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])

def load_img(path, ch=3):
    """Load image → float32 numpy [0,1] shape (H,W,C)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff') and TIFF_OK:
        try:
            img = tifffile.imread(path).astype(np.float32)
            if img.ndim == 2:   img = np.stack([img]*ch, -1)
            elif img.ndim == 3 and img.shape[0] <= 13:
                img = np.transpose(img, (1,2,0))
            img = img[:, :, :ch]
            if img.shape[2] < ch:
                img = np.concatenate([img,
                    np.repeat(img[:,:,-1:], ch-img.shape[2], axis=2)], axis=2)
            p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
            return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
        except: pass
    mode = "RGB" if ch == 3 else "L"
    arr  = np.array(Image.open(path).convert(mode)).astype(np.float32) / 255.0
    if arr.ndim == 2: arr = np.stack([arr]*ch, -1)
    return arr

def to_tensor(arr):
    return torch.from_numpy(arr.transpose(2,0,1)).float().unsqueeze(0)   # [1,C,H,W]

def to_numpy(t):
    """Tensor [1,C,H,W] or [C,H,W] → numpy (H,W,C) clipped to [0,1]."""
    arr = t.squeeze(0).cpu().permute(1,2,0).numpy()
    return arr.clip(0, 1)


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
def psnr(p, t):
    mse = F.mse_loss(p.clamp(0,1), t.clamp(0,1)).item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else 100.0

def ssim(p, t, ks=11, C1=1e-4, C2=9e-4):
    pad = ks//2
    mu1 = F.avg_pool2d(p, ks, 1, pad); mu2 = F.avg_pool2d(t, ks, 1, pad)
    s1  = F.avg_pool2d(p*p, ks,1,pad) - mu1**2
    s2  = F.avg_pool2d(t*t, ks,1,pad) - mu2**2
    s12 = F.avg_pool2d(p*t, ks,1,pad) - mu1*mu2
    num = (2*mu1*mu2+C1)*(2*s12+C2)
    den = (mu1**2+mu2**2+C1)*(s1+s2+C2)
    return float((num/den).mean())


# ─────────────────────────────────────────
# CORE INFERENCE FUNCTION
# ─────────────────────────────────────────
@torch.no_grad()
def run_inference(lr_path, hr_path=None):
    """
    Run SR on one LR image.
    Returns dict with tensors and metrics (if HR provided).
    """
    lr_np = load_img(lr_path, IN_CH)
    lr_t  = to_tensor(lr_np).to(DEVICE)           # [1,C,H,W]

    # Super-resolve
    sr_t  = model(lr_t)                            # [1,C,4H,4W]

    # Bicubic baseline (same resolution as SR)
    bic_t = F.interpolate(lr_t, scale_factor=SCALE,
                          mode='bicubic', align_corners=False).clamp(0,1)

    result = {
        "lr_path": lr_path,
        "lr_t":    lr_t.cpu(),
        "sr_t":    sr_t.cpu(),
        "bic_t":   bic_t.cpu(),
    }

    if hr_path and os.path.exists(hr_path):
        hr_np = load_img(hr_path, IN_CH)
        hr_t  = to_tensor(hr_np)                   # [1,C,H,W]

        # Resize HR to match SR output if needed (handles slight size mismatches)
        if hr_t.shape[-2:] != sr_t.cpu().shape[-2:]:
            hr_t = F.interpolate(hr_t, size=sr_t.shape[-2:],
                                  mode='bicubic', align_corners=False).clamp(0,1)

        result["hr_t"]       = hr_t
        result["psnr_bic"]   = psnr(bic_t.cpu(), hr_t)
        result["psnr_sr"]    = psnr(sr_t.cpu(),  hr_t)
        result["ssim_bic"]   = ssim(bic_t.cpu(), hr_t)
        result["ssim_sr"]    = ssim(sr_t.cpu(),  hr_t)
        result["gain_psnr"]  = result["psnr_sr"] - result["psnr_bic"]

    return result


# ─────────────────────────────────────────
# PAIRED DATASET MATCHER
# ─────────────────────────────────────────
def match_pairs(hr_folder, lr_folder):
    """Match HR and LR files by stem name. Returns list of (hr_path, lr_path)."""
    hr_dict = {os.path.splitext(os.path.basename(p))[0]: p
               for p in list_images(hr_folder)}
    lr_dict = {os.path.splitext(os.path.basename(p))[0]: p
               for p in list_images(lr_folder)}
    # Handle DIV2K "0001x4.png" naming (strip x4/X4 suffix)
    lr_clean = {}
    for stem, path in lr_dict.items():
        clean = stem.replace('x4','').replace('X4','')
        lr_clean[clean] = path

    pairs = []
    for stem in sorted(set(hr_dict.keys()) & set(lr_clean.keys())):
        pairs.append((hr_dict[stem], lr_clean[stem]))
    return pairs


# ─────────────────────────────────────────
# FULL DATASET EVALUATION
# ─────────────────────────────────────────
def evaluate_dataset(hr_folder, lr_folder, dataset_name, max_images=None):
    """
    Evaluate all LR/HR pairs in a folder.
    Returns list of per-image result dicts and prints summary.
    """
    pairs = match_pairs(hr_folder, lr_folder)
    if not pairs:
        print(f"  ✗ No paired images found for {dataset_name}")
        return []

    if max_images:
        pairs = pairs[:max_images]

    print(f"\n── {dataset_name}  ({len(pairs)} images) ─────────────────────")

    results       = []
    all_psnr_sr   = []
    all_psnr_bic  = []
    all_ssim_sr   = []
    all_ssim_bic  = []

    for hr_path, lr_path in tqdm(pairs, desc=f"  {dataset_name}"):
        r = run_inference(lr_path, hr_path)
        results.append(r)
        all_psnr_sr.append(r["psnr_sr"])
        all_psnr_bic.append(r["psnr_bic"])
        all_ssim_sr.append(r["ssim_sr"])
        all_ssim_bic.append(r["ssim_bic"])

    avg = lambda lst: sum(lst) / len(lst)

    print(f"\n  {'Metric':<18} {'Bicubic':>10} {'SR (Ours)':>12} {'Gain':>8}")
    print(f"  {'─'*50}")
    print(f"  {'PSNR (dB)':<18} {avg(all_psnr_bic):>10.2f} {avg(all_psnr_sr):>12.2f} {avg(all_psnr_sr)-avg(all_psnr_bic):>+7.2f}")
    print(f"  {'SSIM':<18} {avg(all_ssim_bic):>10.4f} {avg(all_ssim_sr):>12.4f} {avg(all_ssim_sr)-avg(all_ssim_bic):>+7.4f}")
    print(f"  {'─'*50}")

    return results


# ─────────────────────────────────────────
# VISUALIZATION — 4-column grid
# ─────────────────────────────────────────
def visualize(results, dataset_name, n=6, save_name=None):
    """
    Plot n sample results: LR | Bicubic | SR (Ours) | HR
    Each column annotated with PSNR/SSIM.
    """
    samples = random.sample(results, min(n, len(results)))
    cols    = ["LR Input", "Bicubic (baseline)", "SR Output (Ours)", "HR Ground Truth"]
    has_hr  = "hr_t" in samples[0]
    n_cols  = 4 if has_hr else 3

    fig, axs = plt.subplots(len(samples), n_cols,
                             figsize=(5.5 * n_cols, 5 * len(samples)))
    if len(samples) == 1:
        axs = [axs]

    fig.suptitle(
        f"Test Results — {dataset_name}\n"
        "LR Input  |  Bicubic Baseline  |  SR Output (Ours)" +
        ("  |  HR Ground Truth" if has_hr else ""),
        fontsize=13, fontweight="bold", y=1.01
    )

    cmap = "gray" if IN_CH == 1 else None

    for row, r in enumerate(samples):
        lr_np  = to_numpy(r["lr_t"])
        sr_np  = to_numpy(r["sr_t"])
        bic_np = to_numpy(r["bic_t"])
        fname  = os.path.basename(r["lr_path"])

        # Column 0 — LR
        axs[row][0].imshow(lr_np, cmap=cmap)
        axs[row][0].set_title("LR Input", fontsize=10, fontweight="bold")
        axs[row][0].set_xlabel(
            f"{lr_np.shape[1]}×{lr_np.shape[0]}px\n{fname}", fontsize=8
        )

        # Column 1 — Bicubic
        axs[row][1].imshow(bic_np, cmap=cmap)
        axs[row][1].set_title("Bicubic", fontsize=10)
        if has_hr:
            axs[row][1].set_xlabel(
                f"PSNR: {r['psnr_bic']:.2f} dB\nSSIM: {r['ssim_bic']:.4f}", fontsize=8
            )

        # Column 2 — SR (ours)
        axs[row][2].imshow(sr_np, cmap=cmap)
        axs[row][2].set_title("SR (Ours) ★", fontsize=10, color="green", fontweight="bold")
        if has_hr:
            gain_col = "green" if r["gain_psnr"] >= 0 else "red"
            axs[row][2].set_xlabel(
                f"PSNR: {r['psnr_sr']:.2f} dB  ({r['gain_psnr']:+.2f})\n"
                f"SSIM: {r['ssim_sr']:.4f}",
                fontsize=8, color=gain_col
            )
        else:
            axs[row][2].set_xlabel(
                f"{sr_np.shape[1]}×{sr_np.shape[0]}px (upscaled ×{SCALE})", fontsize=8
            )

        # Column 3 — HR ground truth (if available)
        if has_hr and n_cols == 4:
            hr_np = to_numpy(r["hr_t"])
            axs[row][3].imshow(hr_np, cmap=cmap)
            axs[row][3].set_title("HR Ground Truth", fontsize=10)
            axs[row][3].set_xlabel(
                f"{hr_np.shape[1]}×{hr_np.shape[0]}px", fontsize=8
            )

        for ax in axs[row][:n_cols]:
            ax.axis("off")

    plt.tight_layout()
    fname_out = save_name or f"{dataset_name.lower().replace(' ','_')}_results.png"
    save_path = os.path.join(OUT_DIR, fname_out)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# SECTION A: EVALUATE — DIV2K VALIDATION
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  A. DIV2K Validation Set (standard benchmark)")
print("="*60)

div2k_results = evaluate_dataset(
    hr_folder=DIV_VAL_HR,
    lr_folder=DIV_VAL_LR,
    dataset_name="DIV2K Validation",
    max_images=100    # use all 100 validation images
)
if div2k_results:
    visualize(div2k_results, "DIV2K Validation", n=6,
              save_name="div2k_test_results.png")


# ─────────────────────────────────────────
# SECTION B: EVALUATE — SATELLITE IMAGES
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  B. 4× Satellite Images (HR_0.5m / LR_2m)")
print("="*60)

sat_results = evaluate_dataset(
    hr_folder=SAT_HR,
    lr_folder=SAT_LR,
    dataset_name="Satellite (4× SR)",
    max_images=None   # use all available satellite pairs
)
if sat_results:
    visualize(sat_results, "Satellite SR", n=6,
              save_name="satellite_test_results.png")


# ─────────────────────────────────────────
# SECTION C: PER-IMAGE CSV REPORT
# ─────────────────────────────────────────
all_results = []
for r in div2k_results:
    r["dataset"] = "DIV2K"
    all_results.append(r)
for r in sat_results:
    r["dataset"] = "Satellite"
    all_results.append(r)

csv_path = os.path.join(OUT_DIR, "per_image_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "filename", "psnr_bicubic", "psnr_sr",
                     "ssim_bicubic", "ssim_sr", "gain_psnr"])
    for r in all_results:
        if "psnr_sr" in r:
            writer.writerow([
                r["dataset"],
                os.path.basename(r["lr_path"]),
                f"{r['psnr_bic']:.4f}",
                f"{r['psnr_sr']:.4f}",
                f"{r['ssim_bic']:.4f}",
                f"{r['ssim_sr']:.4f}",
                f"{r['gain_psnr']:.4f}",
            ])
print(f"\n  Per-image CSV saved: {csv_path}")


# ─────────────────────────────────────────
# SECTION D: FINAL SUMMARY TABLE
# ─────────────────────────────────────────
def summary(results, name):
    if not results or "psnr_sr" not in results[0]: return
    ps  = [r["psnr_sr"]  for r in results]
    pb  = [r["psnr_bic"] for r in results]
    ss  = [r["ssim_sr"]  for r in results]
    sb  = [r["ssim_bic"] for r in results]
    avg = lambda l: sum(l)/len(l)
    return {
        "Dataset":        name,
        "N":              len(results),
        "PSNR Bicubic":   f"{avg(pb):.2f}",
        "PSNR SR":        f"{avg(ps):.2f}",
        "PSNR Gain":      f"{avg(ps)-avg(pb):+.2f}",
        "SSIM Bicubic":   f"{avg(sb):.4f}",
        "SSIM SR":        f"{avg(ss):.4f}",
    }

rows = [
    summary(div2k_results, "DIV2K Validation"),
    summary(sat_results,   "Satellite (4×)"),
]
rows = [r for r in rows if r]

print("\n" + "="*70)
print("  FINAL SUMMARY")
print("="*70)
if rows:
    headers = list(rows[0].keys())
    # header row
    print("  " + "  ".join(f"{h:<20}" for h in headers))
    print("  " + "─"*65)
    for row in rows:
        print("  " + "  ".join(f"{row[h]:<20}" for h in headers))
print("="*70)
print(f"\n  All outputs saved to: {OUT_DIR}/")
print("  Files:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"    {f:<40} {size/1024:.1f} KB")
