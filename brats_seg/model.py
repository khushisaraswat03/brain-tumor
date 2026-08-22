import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_channels=8, depth=3):
        super().__init__()
        self.depth = depth

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        widths = [base_channels * (2 ** i) for i in range(depth + 1)]
        for i in range(depth):
            self.enc_blocks.append(ConvBlock(ch, widths[i]))
            self.pools.append(nn.MaxPool3d(2))
            ch = widths[i]

        self.bottleneck = ConvBlock(ch, widths[depth])
        ch = widths[depth]

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            self.upconvs.append(
                nn.ConvTranspose3d(ch, widths[i], kernel_size=2, stride=2)
            )
            self.dec_blocks.append(ConvBlock(widths[i] * 2, widths[i]))
            ch = widths[i]

        self.head = nn.Conv3d(ch, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = self._match_and_cat(x, skip)
            x = dec(x)

        return self.head(x)

    @staticmethod
    def _match_and_cat(x, skip):
        if x.shape[2:] != skip.shape[2:]:
            diffs = [s - t for s, t in zip(skip.shape[2:], x.shape[2:])]
            pad = []
            for d in reversed(diffs):
                lo = d // 2
                hi = d - lo
                pad.extend([lo, hi])
            x = nn.functional.pad(x, pad)
        return torch.cat([x, skip], dim=1)

def build_model(cfg):
    m = cfg.model
    return UNet3D(
        in_channels=m.get("in_channels", 4),
        num_classes=m.get("num_classes", 4),
        base_channels=m.get("base_channels", 8),
        depth=m.get("depth", 3),
    )
