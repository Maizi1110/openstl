import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.layers import DropPath, trunc_normal_

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2],
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B,
                     D // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    return tuple(use_window_size) if shift_size is None else (tuple(use_window_size), tuple(use_shift_size))


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x, attn_mask = x, None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3)) if any(
            i > 0 for i in shift_size) else shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, scale=(1, 2, 2)):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0, x1, x2, x3 = x[:, :, 0::2, 0::2, :], x[:, :, 1::2, 0::2, :], x[:, :, 0::2, 1::2, :], x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return self.reduction(self.norm(x))


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class BasicLayer_skip(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, subsample=None, subsample_scale=(1, 2, 2), use_checkpoint=False):
        super().__init__()
        self.window_size, self.shift_size = window_size, tuple(i // 2 for i in window_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size,
                                   shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)])
        self.subsample = subsample(dim=dim, norm_layer=norm_layer, scale=subsample_scale) if subsample else None

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp, Hp, Wp = [int(np.ceil(s / w)) * w for s, w in zip((D, H, W), window_size)]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks: x = blk(x, attn_mask)
        x_skip = rearrange(x.view(B, D, H, W, -1), 'b d h w c -> b c d h w')
        if self.subsample: x = self.subsample(x.view(B, D, H, W, -1))
        return rearrange(x, 'b d h w c -> b c d h w') if self.subsample else x_skip, x_skip


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _, _, D, H, W = x.size()
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        x = self.proj(F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d)))
        if self.norm:
            B, C, nD, nH, nW = x.shape
            x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, nD, nH, nW)
        return x


class PatchExpanding3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, out_chans=3):
        super().__init__()
        self.deproj = nn.ConvTranspose3d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        try:
            return self.deproj(x)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if ('unable to find an engine' not in msg) and ('cudnn' not in msg):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                try:
                    out = F.conv_transpose3d(
                        x.float(),
                        self.deproj.weight.float(),
                        None if self.deproj.bias is None else self.deproj.bias.float(),
                        stride=self.deproj.stride,
                        padding=self.deproj.padding,
                        output_padding=self.deproj.output_padding,
                        groups=self.deproj.groups,
                        dilation=self.deproj.dilation,
                    )
                except RuntimeError:
                    # Last-resort fallback for drivers/kernels that still reject CUDA deconv.
                    out = F.conv_transpose3d(
                        x.float().cpu(),
                        self.deproj.weight.float().cpu(),
                        None if self.deproj.bias is None else self.deproj.bias.float().cpu(),
                        stride=self.deproj.stride,
                        padding=self.deproj.padding,
                        output_padding=self.deproj.output_padding,
                        groups=self.deproj.groups,
                        dilation=self.deproj.dilation,
                    ).to(device=x.device)
            return out.to(dtype=x.dtype)

class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        B, C, D, H, W = input.size()
        nOut = C // reduce(mul, self.scale)
        input_view = input.contiguous().view(B, nOut, self.scale[0], self.scale[1], self.scale[2], D, H, W)
        return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous().view(B, nOut, D * self.scale[0],
                                                                            H * self.scale[1], W * self.scale[2])


class CubicDualUpsample(nn.Module):
    def __init__(self, dim, scale=(1, 2, 2), norm_layer=nn.LayerNorm):
        super().__init__()
        sf = reduce(mul, scale)
        self.conv_p = nn.Sequential(nn.Conv3d(dim, (sf // 2) * dim, 1, bias=False), nn.PReLU(), PixelShuffle3D(scale),
                                    nn.Conv3d(dim // 2, dim // 2, 1, bias=False))
        self.conv_b = nn.Sequential(nn.Conv3d(dim, dim, 1), nn.PReLU(),
                                    nn.Upsample(scale_factor=scale, mode='trilinear'),
                                    nn.Conv3d(dim, dim // 2, 1, bias=False))
        self.conv_merge = nn.Conv3d(dim, dim // 2, 1, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = rearrange(x, 'B T H W C -> B C T H W')
        out = self.conv_merge(torch.cat([self.conv_p(x), self.conv_b(x)], dim=1))
        x = rearrange(out, 'B C T H W -> B T H W C')
        return self.norm(x)


class exPreCast_Model(nn.Module):
    def __init__(self, in_shape=(13, 1, 112, 112), pre_seq_length=5, aft_seq_length=10,
                 embed_dim=96, depths=[2, 6, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7), mlp_ratio=4., drop_rate=0., drop_path_rate=0.2, **kwargs):
        super().__init__()

        # 闂佽崵濮抽悞锕€顭垮Ο鑲╃鐎广儱顦伴崕搴亜閺冨倹娅曢柛銊ャ偢閹粙顢涢崱妤€顏繛?
        self.output_frames = aft_seq_length
        self.in_chans = in_shape[1]
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(patch_size=(2, 4, 4), in_chans=self.in_chans, embed_dim=embed_dim,
                                        norm_layer=nn.LayerNorm)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder = nn.ModuleList([BasicLayer_skip(int(embed_dim * 2 ** i), depths[i], num_heads[i], window_size,
                                                      mlp_ratio, drop=drop_rate,
                                                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                                      subsample=PatchMerging if i < len(depths) - 1 else None) for i in
                                      range(len(depths))])
        self.bottleneck_upscale = CubicDualUpsample(int(embed_dim * 2 ** (len(depths) - 1)))
        self.decoder = nn.ModuleList([BasicLayer_skip(int(embed_dim * 2 ** i), depths[i], num_heads[i], window_size,
                                                      mlp_ratio, drop=drop_rate,
                                                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                                      subsample=CubicDualUpsample if i > 0 else None) for i in
                                      range(len(depths) - 2, -1, -1)])
        self.patch_expand3d = PatchExpanding3D(patch_size=(2, 4, 4), embed_dim=embed_dim, out_chans=self.in_chans)

        self.last_time_dim = (pre_seq_length + 1) // 2 * 2
        self.time_extractor = nn.Conv3d(self.last_time_dim, aft_seq_length, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)): trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)

        self.apply(_init)

    def forward(self, x, **kwargs):
        # # [DEBUG 闂備浇顫夋禍浠嬪垂娴犲绠い鈽呯秮閺屻劌鈽夊▎鎺戭棟閻熸粍濡搁崶褎宓嶉梺闈浤涚仦钘夊濠电偞鍨堕幐鎾磻閹剧粯鐓欏ù锝呭枤濞兼劖銇勯幇顑惧仮妤犵偛绉归獮姗€宕橀懠顑惧亼
        # debug = not hasattr(self, '_debug_printed')
        # if debug:
        #     print(f"\n婵☆偓绲介崯浼村储?[闂備浇顫夋禍浠嬪垂娴犲绠い?闂備礁鎲￠幐鍝ョ矓閹绢喖绠栨繝濠傚椤曢亶鏌ｅΟ鎸庣彧閻忓骏绱曢埀顒侇問閸犳帡宕戦幘鏂ユ斀妞ゆ梻鈷堥崕婊呯磼鏉堛劌绗氭繛鐓庣箻楠炴﹢宕橀幓鎺撹緢闁荤喐绮忛崺鍥垂缂佹ɑ鍙? {x.shape}闂備焦瀵х粙鎴︽儔閻撳篃鐑樺閺夋垵鍞ㄩ梺鎼炲労娴滆泛煤閿濆绾ч柛顐ゅ枑鐏忣參鏌? {self.output_frames} 闂?)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.patch_embed(x)
        x_skips = []
        for i, layer in enumerate(self.encoder):
            x, skip = layer(x.contiguous())
            if i < self.num_layers - 1: x_skips.append(skip)

        x = self.bottleneck_upscale(rearrange(x, 'n c d h w -> n d h w c'))
        x = rearrange(x, 'n d h w c -> n c d h w')

        for i, layer in enumerate(self.decoder):
            skip = x_skips[-(i + 1)]
            if x.shape != skip.shape: x = x[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]
            x = x + skip
            x, _ = layer(x.contiguous())

        x = self.patch_expand3d(x)

        # 1. 婵犳鍠楃缓鍧楀磹閺嵮屾富闁稿瞼鍋涚粻锝夋煙鐎涙鐭嬬紒顕嗙畵閹兘寮村鍐插帯濠电姭鍋?x 闂備焦鐪归崝宀€鈧凹鍓熼幊婊勫鐎涙ê浜?(B, C, T, H, W)
        # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞鐎ｎ偂姘﹀┑鐐叉閹哥鈻?T 闂佽绻愮换鎰暦椤掑嫬绀勯柨娑樺绾惧ジ鐓崶銊︹拹缁?9 濠电偛鐡ㄧ划灞轿涚€靛憡顫曟繝闈涚墛鐎氭岸鎮楀☉娅虫垿锝為弽顓熺厸?U-Net 缂傚倸鍊烽悞锕傚箰婵犳碍鍊垫い鏍ㄥ閸嬫挻鎷呴崘顭戞闂佹悶浼囬崶褏鐫勯梺鍓插亝濞叉﹢鏁嶉悢鐓庣骇闁绘劕鐡ㄧ紞鎴炪亜閵堝懎鏆ｇ€规洘绻堥幃鈺傛綇閳轰焦娅?

        x = rearrange(x, 'B C T H W -> B T H W C')
        x = self.time_extractor(x)
        x = rearrange(x, 'B T H W C -> B C T H W')

        out = x.permute(0, 2, 1, 3, 4).contiguous()

        # 濠电偞鍨堕幐鎼侇敄婢跺鐒藉ù鍏兼綑閸欏﹥銇勯弽銊ф噮缂佸鍨甸埥澶愬棘濞嗘儳鍓版繝鈷€鍛ｇ紒鍌涘浮閺佹劙宕堕埡浣轰化闂傚倷鐒﹁ぐ鍐儔閻撳簶鏋旈柟杈剧畱缁狅絾銇勮箛鎾村櫤闁绘帊绮欏娲敃閿濆棭娼＄紓渚€顤傞崑濠囧极瀹ュ閱囨繝闈涙椤斿姊洪悡搴ｆ瀮濠殿喓鍊濋幆鈧柛娑樼摠閸嬨劑鏌ｉ弬鎸庡暈婵炲懎绻橀弻?
        if out.shape[1] > self.output_frames:
            out = out[:, :self.output_frames, ...].contiguous()

        return out