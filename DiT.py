# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional
import random
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np
from attention import MultiheadAttention
from transformer.Layers import EncoderLayer, DecoderLayer

import math


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None   
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2



def position_aware_average_pooling(video_features, position_features):
    # video_features: (batch_size, frames, dim)  
    # position_features: (batch_size, num_position, 2)   
    
    batch_size, frames, dim = video_features.size()
    num_position = position_features.size(1)
    
    true_position = position_features * frames
    weights = torch.zeros(batch_size, frames, num_position).cuda()

    center = true_position[:, :, 0] #32 10 1
    length = true_position[:, :, 1]
    start_frame = torch.clamp(torch.round(center - length / 2), min=0, max=frames - 1).long()
    end_frame = torch.clamp(torch.round(center + length / 2), min=0, max=frames - 1).long()

    for i in range(num_position):
        for j in range(batch_size):
            weights[j, start_frame[j,i].long() : end_frame[j,i].long()+1 , i] =  1.0 / (end_frame[j,i].long() - start_frame[j,i].long() + 1)

    pooled_features =  torch.bmm(video_features.permute(0, 2, 1), weights).permute(2, 0, 1) 
    return pooled_features


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings).cuda()
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=5, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_class=2,
                 scale=2.,
                 denoise_network=None,
                 steps = 50
                 ):
        super().__init__()

        # normalize_before transformer前是否进行norm，默认 false
        self.num_class = num_class

        t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before) # decoder 层
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm) # 多层 decoder 层

        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before) # encoder 层
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm) # 多层 encoder 层

        self.d_model = d_model # 模型维度
        self.nhead = nhead # 多头注意力头数
        self.dec_layers = num_decoder_layers # 降噪网络的 decoder 层数
        # self.num_queries = num_queries
        self.num_patterns = num_patterns # 默认 0


        #diff layer
        diff_layer = DiffDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        # self.diffusion_decoder = DiffDecoder_VQADiff(diff_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec,
        #                                   d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
        #                                   modulate_t_attn=modulate_t_attn,
        #                                   bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
        #                                   num_class=num_class) # 降噪网络中的 decoder 层
        self.denoise_network = denoise_network # 降噪网络中的 decoder 层


        # build diffusion
        timesteps = steps # 训练步数 
        sampling_timesteps = steps  # 推理（采样）步数
        # sampling_timesteps = 50 
        self.num_proposals = num_queries

        betas = cosine_beta_schedule(timesteps)
        # betas =linear_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.scale =  scale
        self.span_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas) # 不更新，可随模型保存
        # self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # self.diff_span_embed = MLP(d_model, d_model, num_class, 1)
        self.diff_span_embed = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_class)
            )
        self.diff_class_embed = nn.Linear(d_model, 2)  # 0: background, 1: foreground
        self._reset_parameters()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x_spans, condition, time_cond):
        x_spans = torch.clamp(x_spans, min=-1 * self.scale, max=self.scale)
        x_spans = ((x_spans / self.scale) + 1) / 2
        # print(x_spans.shape, condition.shape, time_cond.shape)
        main_fea = self.denoise_network(x_spans, condition, time_cond) # b, d
        # import pdb; pdb.set_trace()
        main_cood = self.diff_span_embed(main_fea) # b, n
        # main_cood = main_cood.sigmoid()

        x_start = main_cood  # (batch, num_proposals, 2)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x_spans, time_cond, x_start)

        return pred_noise, x_start, main_fea, main_cood

    @torch.no_grad()
    def ddim_sample(self, batched_moments):
        if isinstance(batched_moments, tuple):
            batch = batched_moments[0].shape[0]
        else:
            batch = batched_moments.shape[0] # b
        shape = (batch, self.num_class) # b, n
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        x_spans = torch.randn(shape).cuda() # b, n 
        # import pdb;pdb.set_trace()
        step_num = 0

        all_hs = [x_spans]

        for time, time_next in time_pairs: #T-1->T-2 .....

            step_num = step_num+1

            time_cond = torch.full((batch,), time).cuda().long() # b
            pred_noise, pred_start, hs, hs_cood = self.model_predictions(x_spans, batched_moments, 
                                                        time_cond)
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            cc = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(pred_start)   
            x_spans = pred_start * alpha_next.sqrt() + cc * pred_noise + sigma * noise # b, n 
            # if time == 24 or time == 0:
            #     print(torch.topk(hs_cood, 8)[1][0])
            #     print(F.softmax(torch.topk(hs_cood, 8)[0][0], dim=-1))
                # import pdb;pdb.set_trace()
            all_hs.append(hs_cood)
        # hs_class = self.diff_class_embed(hs) 
        # all_hs = torch.stack(all_hs, dim=1).squeeze(2)
        # topk_i = [j.item() for j in torch.topk(hs_cood,8)[1][0]]
        # print(topk_i)
        # for i, hs in enumerate(all_hs):
        #     print(i, F.softmax(hs[0][topk_i], dim=-1))
        return hs_cood # b, n

    # @torch.no_grad()
    # def p_sample(self, batched_moments):
    #     if isinstance(batched_moments, tuple):
    #         batch = batched_moments[0].shape[0]
    #     else:
    #         batch = batched_moments.shape[0] # b
    #     shape = (batch, self.num_class) # b, n
    #     total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     x_start = None
    #     x_spans = torch.randn(shape).cuda() # b, n 
    #     step_num = 0

    #     for time, time_next in time_pairs: #T-1->T-2 .....

    #         step_num = step_num+1

    #         time_cond = torch.full((batch,), time).cuda().long() # b
    #         pred_noise, pred_start, hs, hs_cood = self.model_predictions(x_spans, batched_moments, 
    #                                                     time_cond)
            
    #         alpha = self.alphas[time]
    #         alpha_next = self.alphas_cumprod[time]
    #         beta = self.betas[time]
    #         # sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         # cc = (1 - alpha_next - sigma ** 2).sqrt()
    #         noise = torch.randn_like(pred_start)   
    #         # x_spans = pred_start * alpha_next.sqrt() + cc * pred_noise + sigma * noise # b, n 
    #         x_spans = 1 / torch.sqrt(alpha) * (
    #             x_spans - ((1-alpha) / (torch.sqrt(1-alpha_next))) * pred_noise) + torch.sqrt(
    #                 beta) * noise

    #     # hs_class = self.diff_class_embed(hs) 
    #     return hs_cood # b, n
            
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        # n, 2     1     n, 2
        # b, n     b     b, n
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.triu(noise)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



    def prepare_diffusion_concat(self, x_start):
        """
        param gt_boxes: (center, long) # b, n
        """
        time = torch.randint(0, self.num_timesteps, (x_start.size(0),)).long().cuda() # 随机时间步 b
        noise = torch.randn_like(x_start.float()).cuda() # 随机噪声  b, n

        x_start = (x_start * 2. - 1.) * self.scale # 归一化 [0, 1] --> [-2, 2]
        # noise sample
        x_t = self.q_sample(x_start=x_start, t=time, noise=noise) # 加噪 # b, n
        x_t = torch.clamp(x_t, min= -1 * self.scale, max= self.scale) # 限制 [-2, 2]
        x_t = ((x_t / self.scale) + 1) / 2. # 归一化回来 [-2, 2] --> [0, 1] 

        return x_t, noise, time


    def prepare_targets(self, targets):
        # b, n
        d_spans, d_noise, d_t = self.prepare_diffusion_concat(targets) # 得到 x_t, noise, t   # b, d
        # b, n    b, n    b
        return d_spans,  d_noise, d_t


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, condition, gt_target):
        """
        Args:
            condition: (batch_size, L, d)
            gt_target: (b, d)

        Returns:

        """

        if gt_target is not None:
            ### span normalization
            x_t_spans, noises, time_t = self.prepare_targets(gt_target) # b, d   b, d    b
            # time_t = time_t.squeeze(-1) # b
            ### Span embedding + Intensity-aware attention
            hs = self.denoise_network(x_t_spans, condition, time_t) # b, d
            ### out
            # ### predicted spans
            # hs_class = self.diff_class_embed(hs) # linear # 1, b, 2
            # ### scores
            hs_class = self.diff_span_embed(hs) # MLP # b, n
            # outputs_coord = hs_cood.sigmoid()
        else:
            hs_class = self.ddim_sample(condition) # b, n
            # hs_class, outputs_coord, hs = self.ddim_sample(tgt, memory_local, txt_src, memory_key_padding_mask=mask_local, pos=pos_embed_local)

        # memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d) condition

        return hs_class

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers) # 多层 的 encoder layer
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_new = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        
        assert video_length is not None
        
        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos) # 加入 position embedding

        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:] # 取出 q, k, v == v, q, q
        
        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:, video_length + 1:].unsqueeze(1) # 取出 mask
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1) # 注意力 mask
        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0] # 交叉注意力
        src2 = src[1:video_length + 1] + self.dropout1(src2) # 残差链接
        src3 = self.norm1(src2) # layernorm


        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3)))) # FFN
        src2 = src2 + self.dropout2(src3) # 残差连接
        src2 = self.norm2(src2) # layernorm
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]]) # 连接回去 token, video*, query
        # print('after src shape :',src.shape)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        print('before src shape :', src.shape)
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]

        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1,0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        print('after src shape :',src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0] # 自注意力

        
        src2=torch.where(torch.isnan(src2),torch.full_like(src2,0),src2)

        src = src + self.dropout1(src2) # 残差链接
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # FFN
        src = src + self.dropout2(src2) # 残差链接
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DiffDecoder_MomentDiff(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 num_class=2
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for i in range(num_layers)])
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
        self.ref_point_head = MLP(d_model, d_model, d_model, 1)
        # self.time_mlp = SinusoidalPositionEmbeddings(d_model)
        self.time_mlp = SinusoidalPositionEmbeddings(d_model)

        # 4000 --> d
        self.noise_embed = MLP(num_class,d_model, d_model, 1)
        from transformer.Layers import DecoderLayer_woSA
        self.diff_decoders = nn.ModuleList([DecoderLayer_woSA(d_model, d_model, 1, d_model, d_model) for i in range(1)])

    def forward(self, span, condition, time
                ):
        # 输入 span 噪声数据 b, b
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        time_emb = self.time_mlp(time) # b, d
        span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        ### Span embedding
        # output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb.unsqueeze(1).repeat(1,span.shape[1],1) # b, n, d
        output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb # b, d
        output = output.unsqueeze(1) # b, 1, d

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, condition)

        return output.squeeze(1) # b, d

class DiffDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        
        # self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = MLP(d_model, d_model, d_model, 3)

        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,time_emb,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):
        # tgt: n, b, d span --> b, d
        # memory: l, b , d fusion
        tgt = tgt.unsqueeze(0)
        ### denoiser network
        nr_boxes, N = tgt.shape[0], tgt.shape[1]
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        ### Intensity-aware attention.
        q_content = self.ca_qcontent_proj(tgt) # linear # b, d
        k_content = self.ca_kcontent_proj(memory) # linear # b, l, d
        v = self.ca_v_proj(memory) # linear # b, l ,d

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos) # MLP

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos) # MLP
            q = q_content + q_pos # 相加
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        ### cross attention
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed) # linear
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2) # 连接 pos
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2) # 连接 pos

        # cross attention
        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2) # residual
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # FFN
        tgt = tgt + self.dropout3(tgt2) # residual
        tgt = self.norm3(tgt) 
        return tgt.squeeze(0)



# 深度拷贝，地址不同，互不干扰
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        num_queries=args.num_queries,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    if activation == "elu":
        return F.elu
    if activation == "lrelu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
