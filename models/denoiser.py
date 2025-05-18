import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.layers import to_2tuple, DropPath, trunc_normal_
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from timm.layers import DropPath, trunc_normal_
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import math
# Real-time webcam denoising in Colab (optimized, complete)
import cv2
import torch
import numpy as np
import base64
import io
from PIL import Image as PILImage

class CharbonnierLoss(nn.Module):
    r"""
    Charbonnier Loss (L1 variant of the Huber loss):

        L(x, y) = sqrt( (x - y)^2 + ϵ^2 )

    where ϵ is a small constant for numerical stability and robustness to outliers.
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = 'mean'):
        """
        Args:
            epsilon (float): small constant ε; default 1e-3 as in SwinIR.
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        loss = torch.sqrt(diff * diff + (self.epsilon ** 2))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
# === Window utils ===
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.view(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H//window_size)*(W//window_size)))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.view(B, H, W, -1)

# === Core modules ===
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop_rate=0., proj_drop_rate=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads   = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias
        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*ws[0]-1)*(2*ws[1]-1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws[0]), torch.arange(ws[1]), indexing="ij"))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:,:,None] - coords_flat[:,None,:]
        rel = rel.permute(1,2,0).contiguous()
        rel[:,:,0] += ws[0]-1; rel[:,:,1] += ws[1]-1
        rel[:,:,0] *= 2*ws[1]-1
        idx = rel.sum(-1)
        self.register_buffer("relative_position_index", idx)

        self.qkv       = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads)
        q,k,v = qkv.permute(2,0,3,1,4)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N,N,-1).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.dim        = dim
        self.window_size= to_2tuple(window_size)
        self.shift_size = shift_size
        self.norm1      = nn.LayerNorm(dim)
        self.attn       = WindowAttention(dim, window_size, num_heads,
                                          qkv_bias, attn_drop_rate, drop_rate)
        self.drop_path  = DropPath(drop_path_rate) if drop_path_rate>0 else nn.Identity()
        self.norm2      = nn.LayerNorm(dim)
        self.mlp        = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x, H, W, mask=None):
        B, L, C = x.shape
        assert L==H*W, "Input size mismatch"
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size>0:
            shifted = torch.roll(x, shifts=(-self.shift_size,-self.shift_size), dims=(1,2))
        else:
            shifted = x

        x_windows = window_partition(shifted, self.window_size[0])
        attn_windows = self.attn(x_windows, mask)
        shifted = window_reverse(attn_windows, self.window_size[0], H, W)

        if self.shift_size>0:
            x = torch.roll(shifted, shifts=(self.shift_size,self.shift_size), dims=(1,2))
        else:
            x = shifted

        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

class RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop_rate=0., attn_drop_rate=0., drop_path_rates=0.):
        super().__init__()
        if isinstance(drop_path_rates, list):
            rates = drop_path_rates
        else:
            rates = [drop_path_rates]*depth

        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim, num_heads, window_size,
                shift_size=(window_size//2 if i%2 else 0),
                mlp_ratio=mlp_ratio, qkv_bias=True,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=rates[i]
            ) for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        B,C,H,W = x.shape
        x_flat = x.flatten(2).transpose(1,2)
        for layer in self.layers:
            x_flat = layer(x_flat, H, W)
        x_out = x_flat.transpose(1,2).view(B,C,H,W)
        return x + self.conv(x_out)

class SwinIR(nn.Module):
    def __init__(self,
                 in_chans=3, embed_dim=96,
                 depths=[2,2,2,2], num_heads=[3,3,3,3],
                 window_size=8, mlp_ratio=4.0,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.shallow = nn.Conv2d(in_chans, embed_dim, 3, padding=1)

        total = sum(depths)
        dpr = list(torch.linspace(0, drop_path_rate, total))

        self.rstbs = nn.ModuleList()
        idx = 0
        for d, nh in zip(depths, num_heads):
            self.rstbs.append(RSTB(
                dim=embed_dim, depth=d, num_heads=nh, window_size=window_size,
                mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rates=dpr[idx:idx+d]
            ))
            idx += d

        self.reconstruct = nn.Conv2d(embed_dim, in_chans, 3, padding=1)
        self.apply(self._init_weights)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.rstbs[0].layers[0].window_size[0]  # your window_size

        # 1) Compute padding
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        # 2) Pad on bottom and right
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 3) Your usual shallow + RSTBs
        x0 = self.shallow(x)
        x1 = x0
        for rstb in self.rstbs:
            x1 = rstb(x1)
        out = self.reconstruct(x1) + x

        # 4) Crop back
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

# === DataModule & LightningModule (names checked!) ===

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_ratio=0.85, batch_size=8, num_workers=0):
        super().__init__()
        self.dataset     = dataset
        self.train_ratio = train_ratio
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        total = len(self.dataset)
        t = int(self.train_ratio * total)
        self.train_ds, self.val_ds = random_split(
            self.dataset, [t, total-t],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


class SwinIRLightning(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2,2,2,2),
        num_heads: tuple[int, ...] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        # Save all hyperparameters for logs and checkpointing
        self.save_hyperparameters()

        # Determine number of heads: default to embed_dim//16 per block if not provided
        depths_list = list(self.hparams.depths)
        if self.hparams.num_heads is None:
            default_heads = max(1, embed_dim // 16)
            heads = [default_heads] * len(depths_list)
        else:
            heads = list(self.hparams.num_heads)

        # Instantiate SwinIR with dynamic structure
        self.model = SwinIR(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths_list,
            num_heads=heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Loss and metrics
        self.criterion = CharbonnierLoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        inp = noisy[:, noisy.shape[1]//2]
        out = self(inp)
        loss = self.criterion(out, clean)
        psnr_val = self.psnr(out, clean)
        ssim_val = self.ssim(out, clean)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/psnr", psnr_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ssim", ssim_val, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        inp = noisy[:, noisy.shape[1]//2]
        out = self(inp)
        loss = self.criterion(out, clean)
        psnr_val = self.psnr(out, clean)
        ssim_val = self.ssim(out, clean)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/psnr", psnr_val, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim_val, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # Linear warmup + cosine anneal
        warmup_epochs = 1
        max_epochs = self.trainer.max_epochs
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": "epoch", "frequency": 1}}
