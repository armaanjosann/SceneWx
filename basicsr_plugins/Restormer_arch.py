"""Restormer (Zamir et al., CVPR 2022) without einops, for NAFNet's basicsr (basicsr/models/archs/).

The thesis "light" config (dim 32, blocks [2,3,3,4]) has about 5M parameters. Pads input to a multiple
of 8. Architecture after https://github.com/swz30/Restormer (MIT licence).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Channel LayerNorm on NCHW tensors (Restormer 'WithBias' variant)."""
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        x = x * self.weight[None, :, None, None]
        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x


class FeedForward(nn.Module):
    """Gated-Dconv feed-forward network (GDFN)."""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Multi-Dconv-head transposed attention (MDTA): attention over channels, linear in pixels."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        hd = self.num_heads
        q = q.reshape(b, hd, c // hd, h * w)
        k = k.reshape(b, hd, c // hd, h * w)
        v = v.reshape(b, hd, c // hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature      # b, heads, c/h, c/h
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, ln_bias):
        super().__init__()
        self.norm1 = LayerNorm(dim, ln_bias)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, ln_bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        ln_bias = LayerNorm_type == 'WithBias'
        blk = lambda d, n, h: nn.Sequential(*[TransformerBlock(d, h, ffn_expansion_factor, bias, ln_bias) for _ in range(n)])
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)
        self.encoder_level1 = blk(dim, num_blocks[0], heads[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blk(dim * 2, num_blocks[1], heads[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blk(dim * 4, num_blocks[2], heads[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = blk(dim * 8, num_blocks[3], heads[3])
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = blk(dim * 4, num_blocks[2], heads[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = blk(dim * 2, num_blocks[1], heads[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = blk(dim * 2, num_blocks[0], heads[0])
        self.refinement = blk(dim * 2, num_refinement_blocks, heads[0])
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

    def forward(self, inp):
        H, W = inp.shape[-2:]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        x = F.pad(inp, (0, pw, 0, ph), mode='reflect') if (ph or pw) else inp
        e1 = self.encoder_level1(self.patch_embed(x))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        lat = self.latent(self.down3_4(e3))
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
        out = self.output(self.refinement(d1)) + x
        return out[..., :H, :W]


if __name__ == '__main__':
    net = Restormer(dim=32, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2)
    print('params (M):', sum(p.numel() for p in net.parameters()) / 1e6)
    y = net(torch.rand(1, 3, 100, 132))
    print('out', tuple(y.shape))
