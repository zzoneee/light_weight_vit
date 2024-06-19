import math
import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
# from ..registry import BACKBONES
from timm.models.registry import register_model

import torch.nn.functional as F


class ConvStem(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim,
                              kernel_size=3,
                              stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(self.stem(x))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., num_tokens=49):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        self.num_tokens = num_tokens

    def forward(self, x):
        # global_token, x = x[:, :self.num_tokens].contiguous(), x[:, self.num_tokens:].contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        # x = torch.cat([global_token, x], dim=1)
        return x


def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class Local_Window_SA(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 window_size=2,
                 backbone=False):
        super().__init__()

        self.ws = window_size
        self.heads = num_heads
        self.dim = dim
        self.attn_area = window_size * window_size
        self.backbone = backbone

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        # self.proj = nn.Linear(self.dim, self.dim)

        if backbone:
            # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            # Get pair-wise relative position index for each token inside the window
            self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                        window_size).view(-1))
            # Init relative positional bias
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        x = self.local_window_sa(x)
        return x

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def local_window_sa(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.heads, self.dim // self.heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws

        if self.backbone:
            pos_bias = self._get_relative_positional_bias()
            attn = (attn + pos_bias).softmax(dim=-1)

        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.dim)

        # x = self.proj(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.dwconv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False)
        # self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn1 = nn.LayerNorm(out_planes)
        self.relu1 = nn.ReLU(inplace=True)

        # self.pwconv = nn.Conv2d(out_planes, out_planes, 1, 1, 0, bias=False)
        # # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.bn2 = nn.LayerNorm(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dwconv(x)
        # x = self.bn1(x)
        x = self.bn1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.relu1(x)

        # x = self.pwconv(x)
        # # x = self.bn2(x)
        # x = self.bn2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x = self.relu2(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


def hydra(q, k, v):
    """ Hydra Attention

    Paper link: https://arxiv.org/pdf/2209.07484.pdf (Hydra Attention: Efficient Attention with Many Heads)

    Args:
        q, k, and v should all be tensors of shape
            [batch, tokens, features]
    """
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    kv = (k * v).sum(dim=-2, keepdim=True)
    out = q * kv
    return out


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 num_downs=0,
                 window_size=2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size_backbone=2,
                 use_sa_global=True,
                 num_tokens=49,
                 add_global_rate=0.5,
                 use_proj_global=False,
                 kernel_size=3,
                 hydra=False):
        super().__init__()

        self.window_size = window_size
        self.window_size_backbone = window_size_backbone
        self.num_heads = num_heads
        self.dim = dim
        self.num_downs = num_downs
        self.use_sa_global = use_sa_global
        self.num_tokens = num_tokens
        self.hydra = hydra

        self.use_proj_global = use_proj_global

        if not self.use_sa_global:
            self.add_global_rate = add_global_rate

        # self.norm_down_sampling = [norm_layer(dim) for i in range(num_downs)]
        # self.act_down_sampling = [act_layer() for i in range(num_downs)]
        if num_downs >= 1:
            # self.norm_1_down_sampling = norm_layer(dim)
            # self.act_1_down_sampling = act_layer()
            self.down_1_sampling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        if num_downs >= 2:
            # self.norm_2_down_sampling = norm_layer(dim)
            # self.act_2_down_sampling = act_layer()
            self.down_2_sampling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if num_downs > 0:
            ws = [window_size_backbone]
            for i in range(num_downs - 1):
                ws.append(window_size)

            self.wsas = nn.ModuleList([
                # Local_Window_SA(
                #     dim=dim,
                #     num_heads=num_heads,
                #     qkv_bias=qkv_bias,
                #     qk_scale=qk_scale,
                #     window_size=ws[i],
                #     backbone=(i == 0)) for i in range(num_downs)

                # dim, drop_path = 0., layer_scale_init_value = 1e-6, expan_ratio = 4, kernel_size = 7

                # ConvBNReLU(dim, dim, kernel_size, 1, groups=1) for i in range(num_downs)
                ConvEncoder(dim=dim, kernel_size=kernel_size) if i == 0
                else ConvBNReLU(dim, dim, 3, 1, groups=1)
                for i in range(num_downs)
            ])

        # if num_downs == 0:
        #     self.num_downs_0 = ConvBNReLU(dim, dim, 3, 1, 1)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        self.ga_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if self.use_sa_global:
            self.ug_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.ug_q = nn.Linear(dim, dim, bias=qkv_bias)

        if self.use_proj_global:
            self.proj_global = nn.Linear(dim * 2, dim)

        self.gb_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.gb_q = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.Global_Proj = nn.Linear(dim, dim)
        # self.Global_Proj_Mix = nn.Linear(self.num_tokens, self.num_tokens)
        self.Global_ACT = nn.GELU()
        self.Global_Proj2 = nn.Linear(dim, dim)

        # self.DWConv = nn.Conv2d(dim,
        #                         dim,
        #                         kernel_size=3,
        #                         stride=2,
        #                         padding=0,
        #                         groups=dim,
        #                         bias=False)

    def forward(self, x, global_token, H, W):
        # global_token, x = x[:, :self.num_tokens].contiguous(), x[:, self.num_tokens:].contiguous()
        x_origin = x
        H_dst = H
        W_dst = W
        if self.num_downs > 0:
            B, N_origin, C = x.shape

            # x_out = None
            # x_origin = x
            up_sampling_cnt = 0
            down_sampling_cnt = 0
            x_multi_scale = []

            _, N, _ = x.shape
            H = W = int(N ** 0.5)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            for wsa in self.wsas:
                # _, N, _ = x.shape
                # H = W = int(N ** 0.5)
                # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

                x = wsa(x)

                if len(x_multi_scale) == 0:
                    # x_multi_scale.append(x)
                    x_multi_scale.append(x.reshape(B, C, N).permute(0, 2, 1))

                if self.window_size > 1:
                    # x_ = self.act_down_sampling[down_sampling_cnt](self.norm_down_sampling[down_sampling_cnt](x)).permute(0, 3, 1, 2)
                    if down_sampling_cnt == 0:
                        # x_ = self.act_1_down_sampling(self.norm_1_down_sampling(x)).permute(0, 3, 1, 2)
                        # x = self.down_1_sampling(x_).reshape(B, C, -1).permute(0, 2, 1)

                        x = self.down_1_sampling(x)
                    else:
                        # x_ = self.act_2_down_sampling(self.norm_2_down_sampling(x)).permute(0, 3, 1, 2)
                        # x = self.down_2_sampling(x_).reshape(B, C, -1).permute(0, 2, 1)

                        x = self.down_2_sampling(x)
                    # x_ = x.permute(0, 3, 1, 2)
                    down_sampling_cnt += 1
                    # x = self.down_sampling(x_).reshape(B, C, -1).permute(0, 2, 1)

            # x = self.DWConv(x)

            _, _, H_dst, W_dst = x.shape

            x = self.forward_global_aggregation(x.reshape(B, C, -1).permute(0, 2, 1))

            x_downsampleToken = x
            x_local = x_multi_scale[0]
        else:
            x_downsampleToken = self.forward_global_aggregation(x)
            x_local = x_downsampleToken

            # B, N, C = x.shape
            # H = W = int(N ** 0.5)
            # x = self.num_downs_0(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
            #
            # x_local = x.reshape(B, C, -1).permute(0, 2, 1)

            # x = self.DWConv(x)
            # x_downsampleToken = self.forward_global_aggregation(x.reshape(B, C, -1).permute(0, 2, 1))


        # update global_token
        if self.use_sa_global:
            global_token = self.forward_updata_globalToken(global_token, x_downsampleToken)
        elif self.use_proj_global:
            global_token = self.proj_global(torch.cat([global_token, x_downsampleToken], dim=2))
        else:
            global_token = self.Global_Proj(global_token)
            # global_token = self.Global_Proj_Mix(global_token.permute(0, 2, 1)).permute(0, 2, 1)
            global_token = self.Global_ACT(global_token)
            global_token = self.Global_Proj2(global_token)

            B, _, C = x_downsampleToken.shape
            x_downsampleToken = x_downsampleToken.reshape(B, H_dst, W_dst, C).permute(0, 3, 1, 2)
            x_downsampleToken = F.interpolate(x_downsampleToken, size=(4, 4), mode='bicubic')
            x_downsampleToken = x_downsampleToken.permute(0, 2, 3, 1).reshape(B, -1, C)

            global_token = self.add_global_rate * global_token + (1 - self.add_global_rate) * x_downsampleToken

        x_global = self.forward_global_broadcast(x_origin, global_token)

        x_out = x_local + x_global
        # x_out = torch.cat([global_token, x_out], dim=1)
        x_out = self.proj(x_out)

        return x_out, global_token

    def forward_global_aggregation(self, x):
        """
        q: global tokens (7)
        k: global tokens (7)
        v: global tokens (7)
        """
        # x = x + self.pos_embed
        B, N, C = x.shape
        qkv = self.ga_qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        B, _, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.ga_proj(x)
        return x

    def forward_updata_globalToken(self, global_token, x_downsampleToken):
        """
        q: image tokens (global_token)
        k: global tokens (x_downsampleToken)
        v: global tokens (x_downsampleToken)
        """
        B, N, C = x_downsampleToken.shape
        kv = self.ug_kv(x_downsampleToken)
        k, v = kv.view(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        B, N, C = global_token.shape
        q = self.ug_q(global_token)
        q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.gb_proj(x)
        return x

    def forward_global_broadcast(self, x_origin, global_token):
        """
        q: image tokens (56)
        k: global tokens (7)
        v: global tokens (7)
        """
        if not self.hydra:
            B, N, C = global_token.shape
            kv = self.gb_kv(global_token)
            k, v = kv.view(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

            B, N, C = x_origin.shape
            q = self.gb_q(x_origin)
            q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            B, num_heads, N, _ = q.shape
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            # x = self.gb_proj(x)
            return x
        else:
            B, N, C = global_token.shape
            kv = self.gb_kv(global_token)
            k, v = kv.view(B, N, 2, self.dim).permute(2, 0, 1, 3).unbind(0)

            B, N, C = x_origin.shape
            q = self.gb_q(x_origin)
            q = q.view(B, N, self.dim)

            B, N, C = q.shape
            x = hydra(q, k, v)
            x = x.reshape(B, N, C)

            # x = self.gb_proj(x)
            return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, use_sa_global=True, num_tokens=1, add_global_rate=0.5, window_size=2,
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attention=Attention, qk_scale=None, num_downs=0, window_size_backbone=2,
                 use_proj_global=False, kernel_size=3, hydra=False):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm1 = norm_layer(dim)
        self.normGT = norm_layer(dim)
        # self.attn = attention(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size,
        #                       qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              num_downs=num_downs,
                              window_size=window_size,
                              window_size_backbone=window_size_backbone,
                              act_layer=nn.GELU,
                              norm_layer=nn.LayerNorm,
                              use_sa_global=use_sa_global,
                              num_tokens=num_tokens,
                              add_global_rate=add_global_rate,
                              use_proj_global=use_proj_global,
                              kernel_size=kernel_size,
                              hydra=hydra)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path_global_token = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = BiAttnMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, global_token, H, W):
        # x = x + self.drop_path(self.attn(self.norm1(x), global_token))
        # x = x + self.drop_path(self.mlp(self.norm2(x), global_token))

        x_, global_token_ = self.attn(self.norm1(x), self.normGT(global_token), H, W)

        x_tmp = self.drop_path(torch.cat((global_token_, x_), dim=1))
        global_token_, x_ = x_tmp[:, :self.num_tokens].contiguous(), x_tmp[:, self.num_tokens:].contiguous()
        global_token = global_token + global_token_
        x = x + x_

        # x = x + self.drop_path(x_)
        # # global_token = global_token + self.drop_path_global_token(global_token_)
        # global_token = global_token + global_token_

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, global_token


class ResidualMergePatch(nn.Module):
    def __init__(self, dim, out_dim, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim, bias=False)
        # use MaxPool3d to avoid permutations
        self.maxp = nn.MaxPool3d((2, 2, 1), (2, 2, 1))
        self.res_proj = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x, global_token, H, W):
        # global_token, x = x[:, :self.num_tokens].contiguous(), x[:, self.num_tokens:].contiguous()
        B, L, C = x.shape

        x = x.view(B, H, W, C)
        res = self.res_proj(self.maxp(x).view(B, -1, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x + res
        global_token = self.proj(self.norm2(global_token))
        # x = torch.cat([global_token, x], 1)
        return x, global_token, (H // 2, W // 2)


# @BACKBONES.register_module
class LightViT_DualToken_convEdgenext_mlp_normGT_gt16(nn.Module):

    def __init__(self, img_size=224, patch_size=8, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_layers=[2, 6, 6], num_heads=[2, 4, 8], mlp_ratios=[8, 4, 4], use_sa_global=True, num_tokens=8,
                 add_global_rate=0.5, window_size=2, neck_dim=1280, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=ConvStem, norm_layer=None, act_layer=None, weight_init='',
                 num_downs=[2, 1, 0], window_size_backbone=2, use_proj_global=False, ks=[3, 3, -1],
                 hydras=[False, False, False]):
        super().__init__()

        if use_proj_global:
            use_sa_global = False

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_tokens = num_tokens
        self.mlp_ratios = mlp_ratios
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.window_size = window_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])

        self.global_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dims[0]))

        stages = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]  # stochastic depth decay rule
        # self.pos_embeds = []
        for stage, (embed_dim, num_layer, num_head, mlp_ratio, kernel_size, hydra) in enumerate(
                zip(embed_dims, num_layers, num_heads, mlp_ratios, ks, hydras)):

            # num_patches = (img_size // patch_size // (2 ** stage)) ** 2
            # self.pos_embeds.append(nn.Parameter(torch.zeros(1, num_patches, embed_dims[stage])))

            blocks = []
            if stage > 0:
                # downsample
                blocks.append(ResidualMergePatch(embed_dims[stage - 1], embed_dim, num_tokens=num_tokens))
            blocks += [
                Block(
                    dim=embed_dim, num_heads=num_head, use_sa_global=use_sa_global, num_tokens=num_tokens,
                    add_global_rate=add_global_rate, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(num_layers[:stage]) + i],
                    norm_layer=norm_layer, act_layer=act_layer, attention=Attention, num_downs=num_downs[stage],
                    window_size_backbone=window_size_backbone, use_proj_global=use_proj_global, kernel_size=kernel_size,
                    hydra=hydra)
                for i in range(num_layer)
            ]
            blocks = nn.Sequential(*blocks)
            stages.append(blocks)
        self.stages = nn.Sequential(*stages)

        self.norm = norm_layer(embed_dim)

        self.neck = nn.Sequential(
            nn.Linear(embed_dim, neck_dim),
            nn.LayerNorm(neck_dim),
            nn.GELU()
        )

        self.head = nn.Linear(neck_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.global_token, std=.02)
            self.apply(_init_vit_weights)

        # for pos_embed in self.pos_embeds:
        #     trunc_normal_(pos_embed, std=.02)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'global_token', '[g]relative_position_bias_table'}

    def forward_features(self, x):
        x, (H, W) = self.patch_embed(x)
        global_token = self.global_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((global_token, x), dim=1)
        # stage_cnt = 0
        for stage in self.stages:
            # if stage_cnt == 0:
            #     stage_cnt += 1
            #     pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            #     x = x + pos_embed
            for block in stage:
                if isinstance(block, ResidualMergePatch):
                    x, global_token, (H, W) = block(x, global_token, H, W)
                    # pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
                    # x = x + pos_embed
                elif isinstance(block, Block):
                    x, global_token = block(x, global_token, H, W)
                else:
                    x, global_token = block(x, global_token)

        x = torch.cat((global_token, x), dim=1)
        x = self.norm(x)
        x = self.neck(x)
        return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# @register_model
# def lightvit_dualToken_left_xtiny_convEdgenext_mix_normGT(pretrained=False, **kwargs):
#     model_kwargs = dict(patch_size=8,
#                         embed_dims=[48, 96, 192],
#                         num_layers=[2, 6, 4],
#                         num_heads=[2, 4, 8],
#                         mlp_ratios=[8, 4, 4],
#                         use_sa_global=False,
#                         num_tokens=49,
#                         add_global_rate=0.1,
#                         # drop_path_rate=0.1,
#                         drop_rate=0.,
#                         window_size_backbone=7,
#                         # global_proj='left',
#                         # global_ffn=False,
#                         ks=[5, 7, -1],
#                         # hydras=[False, False, True],
#                         **kwargs)
#     model = LightViT_DualToken_convEdgenext_mix_normGT(**model_kwargs)
#     return model


@register_model
def lightvit_dualToken_left_tiny_convEdgenext_mlp_normGT_gt16(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=8,
                        embed_dims=[64, 128, 256],
                        num_layers=[2, 6, 6],
                        num_heads=[2, 4, 8],
                        mlp_ratios=[8, 4, 4],
                        use_sa_global=False,
                        num_tokens=16,
                        add_global_rate=0.1,
                        # drop_path_rate=0.1,
                        drop_rate=0.,
                        window_size_backbone=7,
                        # global_proj='left',
                        # global_ffn=False,
                        ks=[5, 7, -1],
                        # hydras=[False, False, True],
                        **kwargs)
    model = LightViT_DualToken_convEdgenext_mlp_normGT_gt16(**model_kwargs)
    return model

# +---------------+--------+
# | operator type | Gflops |
# +---------------+--------+
# |      conv     | 0.058  |
# |   batch_norm  | 0.003  |
# |   layer_norm  | 0.008  |
# |     matmul    | 0.884  |
# |     addmm     | 0.001  |
# +---------------+--------+
# Total flops: 1.0G ± 0.0G
# (ev1) root@iZuf6hijpuiho6kt4lkaehZ:~/EasyCV-features-EdgeVit3# python
# Python 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)
# [GCC 7.5.0] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> 0.058+0.003+0.008+0.884+0.001
# 0.9540000000000001


# Number of backbone parameters: 11.852 M
# Number of backbone parameters requiring grad: 11.852 M
# Number of total parameters: 11.852 M
# Number of total parameters requiring grad: 11.852 M




# @register_model
# def lightvit_dualToken_left_small_convEdgenext_mix_normGT(pretrained=False, **kwargs):
#     model_kwargs = dict(patch_size=8,
#                         embed_dims=[96, 192, 384],
#                         num_layers=[2, 6, 6],
#                         num_heads=[3, 6, 12, ],
#                         mlp_ratios=[8, 4, 4],
#                         use_sa_global=False,
#                         num_tokens=49,
#                         add_global_rate=0.1,
#                         # drop_path_rate=0.1,
#                         drop_rate=0.,
#                         window_size_backbone=7,
#                         # global_proj='left',
#                         # global_ffn=False,
#                         ks=[5, 7, -1],
#                         # hydras=[False, False, True],
#                         **kwargs)
#     model = LightViT_DualToken_convEdgenext_mix_normGT(**model_kwargs)
#     return model
#
# @register_model
# def lightvit_dualToken_left_base_convEdgenext_mix_normGT(pretrained=False, **kwargs):
#     model_kwargs = dict(patch_size=8,
#                         embed_dims=[128, 256, 512],
#                         num_layers=[3, 8, 6],
#                         num_heads=[4, 8, 16, ],
#                         mlp_ratios=[8, 4, 4],
#                         use_sa_global=False,
#                         num_tokens=49,
#                         add_global_rate=0.1,
#                         # drop_path_rate=0.1,
#                         drop_rate=0.,
#                         window_size_backbone=7,
#                         # global_proj='left',
#                         # global_ffn=False,
#                         ks=[5, 7, -1],
#                         # hydras=[False, False, True],
#                         **kwargs)
#     model = LightViT_DualToken_convEdgenext_mix_normGT(**model_kwargs)
#     return model
