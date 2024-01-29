from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat
from torchsummary import summary


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# local + global
class LGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 # new added
                 first_attn_type='self', third_attn_type='cross',
                 attn_param_sharing_first_third=False, attn_param_sharing_all=False,
                 no_second=False, no_third=False,
                 ):

        super().__init__()

        assert first_attn_type in ['self', 'cross'], f"Error: invalid attention type '{first_attn_type}', expected 'self' or 'cross'!"
        assert third_attn_type in ['self', 'cross'], f"Error: invalid attention type '{third_attn_type}', expected 'self' or 'cross'!"
        self.first_attn_type = first_attn_type
        self.third_attn_type = third_attn_type
        self.attn_param_sharing_first_third = attn_param_sharing_first_third
        self.attn_param_sharing_all = attn_param_sharing_all

        # Attention layer
        ## perform local (intra-region) attention, update messenger tokens
        ## (local->messenger) or (local<->local, local<->messenger)
        self.first_attn_norm0 = norm_layer(dim)
        if self.first_attn_type == 'cross':
            self.first_attn_norm1 = norm_layer(dim)
        self.first_attn = GeneralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        ## perform global (inter-region) attention on messenger tokens
        ## (messenger<->messenger)
        self.no_second = no_second
        if not no_second:
            self.second_attn_norm0 = norm_layer(dim)
            if attn_param_sharing_all:
                self.second_attn = self.first_attn
            else:
                self.second_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        ## perform local (intra-region) attention to inject global information into local tokens
        ## (messenger->local) or (local<->local, local<->messenger)
        self.no_third = no_third
        if not no_third:
            self.third_attn_norm0 = norm_layer(dim)
            if self.third_attn_type == 'cross':
                self.third_attn_norm1 = norm_layer(dim)
            if attn_param_sharing_first_third or attn_param_sharing_all:
                self.third_attn = self.first_attn
            else:
                self.third_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        # FFN layer
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None


    def forward(self, x):
        """
        :param x: (B*N, S, C),
            B: batch size
            N: number of local regions
            S: 1 + region size, 1: attached messenger token for each local region
            C: feature dim
        param b: batch size
        :return: (B*N, S, C),
        """
        b = x.size(0) # number of local regions
        if self.gamma_1 is None:
            # Attention layer
            ## perform local (intra-region) self-attention
            if self.first_attn_type == 'self':
                x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))
            else: # 'cross'
                x[:,:1] = x[:,:1] + self.drop_path(
                    self.first_attn(
                        self.first_attn_norm0(x[:,:1]), # (b*n, 1, c)
                        context=self.first_attn_norm1(x[:,1:]) # (b*n, s-1, c)
                    )
                )

            ## perform global (inter-region) self-attention
            if not self.no_second:
                # messenger_tokens: representative tokens
                messenger_tokens = rearrange(x[:,0], 'b c -> b 1 c', b=b).clone() # attn on 'n' dim
                messenger_tokens = messenger_tokens + self.drop_path(
                    self.second_attn(self.second_attn_norm0(messenger_tokens))
                )
                x[:,0] = rearrange(messenger_tokens, 'b 1 c -> b c')
            else: # for usage in the third attn
                messenger_tokens = rearrange(x[:,0], 'b c -> b 1 c') # attn on 'n' dim

            ## perform local-global interaction
            if not self.no_third:
                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))
                else:
                    local_tokens = x[:, 1:].clone()  # NOTE: n merges into s (not b), (B, N*(S-1), D)
                    local_tokens = local_tokens + self.drop_path(
                        self.third_attn(
                            self.third_attn_norm0(local_tokens),  # (b, n*(s-1), c)
                            context=self.third_attn_norm1(messenger_tokens)  # (b, n*1, c)
                        )
                    )
                    x[:,1:] = local_tokens

            # FFN layer
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":

    model = LGBlock(dim=196,num_heads=4)
    model.to('cuda')
    summary(model,(16,196))

