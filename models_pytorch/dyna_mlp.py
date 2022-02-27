import torch
from torch import nn
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange, Reduce

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DynaMixerOp_w(nn.Module):
    def __init__(self, w, dim, hidden_dim, segment):
        super().__init__()
        self.attend = nn.Sequential(
            Rearrange('b h w d -> b d w h'),
            nn.Conv2d(dim, hidden_dim, 1),
            Rearrange('b d w h -> b (d w) h'),
            nn.Conv1d(hidden_dim * w, w * w, 1, groups=segment),
            Rearrange('b (w1 w2) h -> b h w1 w2', w1 = w),
            nn.Softmax(dim = -1),
        )

    def forward(self, x):
        # b h w d = X.shape
        attn = self.attend(x)
        x = torch.matmul(attn, x)
        return x

class DynaMixerOp_h(nn.Module):
    def __init__(self, h, dim, hidden_dim, segment):
        super().__init__()
        self.reshape = Rearrange('b h w d -> b w h d')
        self.attend = nn.Sequential(
            Rearrange('b h w d -> b d h w'),
            nn.Conv2d(dim, hidden_dim, 1),
            Rearrange('b d h w -> b (d h) w'),
            nn.Conv1d(hidden_dim * h, h * h, 1, groups=segment),
            Rearrange('b (h1 h2) w -> b w h1 h2', h1 = h),
            nn.Softmax(dim = -1),
        )
        self.recover = Rearrange('b w h d -> b h w d')

    def forward(self, x):
        # b h w d = X.shape
        attn = self.attend(x)
        x = self.reshape(x)
        x = torch.matmul(attn, x)
        x = self.recover(x)
        return x

class DynaBlock(nn.Module):
    def __init__(self, h, w, dim, hidden_dim_DMO = 2, segment = 8):
        super().__init__()
        self.proj_c = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)

        self.DynaMixerOp_w = DynaMixerOp_w(w, dim, hidden_dim_DMO, segment)
        self.DynaMixerOp_h = DynaMixerOp_h(h, dim, hidden_dim_DMO, segment)

    def forward(self, x):
        Y_c = self.proj_c(x)
        Y_h = self.DynaMixerOp_h(x)
        Y_w = self.DynaMixerOp_w(x)
        Y_out = Y_h + Y_w + Y_c
        Y_out = self.proj_o(Y_out)
        return Y_out

class DynaMLPBlock(nn.Module):
    def __init__(self, depth, h, w, dim, hidden_dim_DMO, segment, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.reshape = Rearrange('b c h w -> b h w c')
        self.recover = Rearrange('b h w c -> b c h w')
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DynaBlock(h, w, dim, hidden_dim_DMO, segment)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = 0.)),
            ]))
    def forward(self, x):
        x = self.reshape(x)
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        x = self.recover(x)
        return x


dynamlp_settings = {
    'T': [[7, 2], [192, 384], [4, 14], [8, 16], 3, 0.1, 2],       # [layers]
    'M': [[7, 2], [256, 512], [7, 17], [8, 16], 3, 0.1, 2],
    'L': [[7, 2], [256, 512], [9, 27], [8, 16], 3, 0.3, 8],
}

class DynaMixer(nn.Module):
    def __init__(self, model_name: str = 'M', image_size = 224, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        assert model_name in dynamlp_settings.keys(), f"DynaMLP model name should be in {list(dynamlp_settings.keys())}"
        patch_size, embed_dims, depths, segment, mlp_ratio, dropout, hidden_dim_DMO = dynamlp_settings[model_name]

        image_height, image_width = pair(image_size)
        h = []
        w = []
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            try:
                h.append(int(h[-1] / ps[0]))
                w.append(int(w[-1] / ps[1]))
            except:
                h.append((int)(image_height / ps[0]))
                w.append((int)(image_width / ps[1]))
            assert (image_height % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_width % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]


        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=patch_size[i], stride=patch_size[i]),
                DynaMLPBlock(depth = depths[i], h = h[i], w = w[i], dim = embed_dims[i], hidden_dim_DMO = hidden_dim_DMO, segment = segment[i], 
                    mlp_dim = embed_dims[i] * mlp_ratio, dropout = dropout)
            ) for i in range(self.stage)]
        )
        
        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dims[-1], num_classes)
        )

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out

