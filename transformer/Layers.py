''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiHeadAttentionPre, PositionwiseFeedForwardPre, MultiHeadAttention_scale, PositionwiseFeedForward_scale


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
    
class EncoderLayerPre(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayerPre, self).__init__()
        self.slf_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
    
    
class DecoderLayerPre(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerPre, self).__init__()
        self.slf_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
    
class DecoderLayer_woSA(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_woSA, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn, dec_enc_attn

class DecoderLayer_scale(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_scale, self).__init__()
        self.enc_attn = MultiHeadAttention_scale(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward_scale(d_model, d_inner, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.ELU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(
            self, dec_input, enc_output, time_embed,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_embed).chunk(6, dim=1)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, shift_msa, scale_msa, gate_msa, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output, shift_mlp, scale_mlp, gate_mlp)
        return dec_output, dec_enc_attn, dec_enc_attn

class DecoderLayerPre_woSA(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerPre_woSA, self).__init__()
        self.enc_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn, dec_enc_attn

class DecoderLayer_Enc(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_Enc, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # b, l1, d b, l2, d
        l1 = dec_input.size(1)
        fusion_input = torch.cat([dec_input, enc_output], dim=1) # b, l1+l2, d
        dec_output, dec_slf_attn = self.slf_attn(
            fusion_input, fusion_input, fusion_input, mask=slf_attn_mask)
        dec_output = dec_output[:, :l1] # b, l1, d
        # dec_output, dec_slf_attn = self.slf_attn(
        #     fusion_input, fusion_input, fusion_input, mask=slf_attn_mask) # b, l1+l2, d
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_slf_attn
    
class DecoderLayerPre_Enc(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerPre_Enc, self).__init__()
        self.slf_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # b, l1, d b, l2, d
        l1 = dec_input.size(1)
        fusion_input = torch.cat([dec_input, enc_output], dim=1) # b, l1+l2, d
        dec_output, dec_slf_attn = self.slf_attn(
            fusion_input, fusion_input, fusion_input, mask=slf_attn_mask)
        dec_output = dec_output[:, :l1] # b, l1, d
        # dec_output, dec_slf_attn = self.slf_attn(
        #     fusion_input, fusion_input, fusion_input, mask=slf_attn_mask) # b, l1+l2, d
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_slf_attn
    


class DecoderLayer_context(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_context, self).__init__()
        self.context_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_input, context,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # import pdb; pdb.set_trace()
        enc_output, dec_slf_attn = self.context_attn(
            enc_input, context, context, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.dec_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class DecoderLayer_multi(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_multi, self).__init__()
        self.context_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_input, context,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # import pdb; pdb.set_trace()
        enc_output, dec_slf_attn = self.context_attn(
            dec_input, enc_input, enc_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.dec_attn(
            enc_output, context, context, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class DecoderLayerPre_context(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerPre_context, self).__init__()
        self.context_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.dec_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_input, context,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # import pdb; pdb.set_trace()
        enc_output, dec_slf_attn = self.context_attn(
            enc_input, context, context, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.dec_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
    
class DecoderLayerPre_context_2(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerPre_context_2, self).__init__()
        self.context_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.dec_attn = MultiHeadAttentionPre(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardPre(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_input, context,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # import pdb; pdb.set_trace()
        enc_output, dec_slf_attn = self.context_attn(
            context, enc_input, enc_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.dec_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
