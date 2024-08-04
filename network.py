import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import random
from transformers import AutoModel
from transformer.Models import PositionalEncoding
from transformer.Layers import EncoderLayer, DecoderLayer
from DiT import Transformer


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DenoiseNetwork(nn.Module):

    def __init__(self, module_dim=768, module_id=0, layer_num=2, num_class=2):
        super().__init__()


        self.module_dim = module_dim
        self.num_class = num_class

        # 时间编码器
        # self.time_embed = SinusoidalPositionEmbeddings(d_model) # diff_mom 正余弦编码
        from DiT_module import TimestepEmbedder
        self.time_embed = TimestepEmbedder(self.module_dim) # DiT 正余弦加mlp
        
        # 噪声编码器
        # 4000 --> d
        # self.noise_embed = MLP(num_class,d_model, d_model, 1) # diff_mom MLP
        self.noise_embed = nn.Linear(self.num_class, self.module_dim) # DiT linear
        from transformer.Layers import DecoderLayer_woSA, DecoderLayerPre_woSA, DecoderLayerPre_Enc, DecoderLayer_Enc, DecoderLayer_context
        if module_id == 0:
            self.diff_decoders = nn.ModuleList([DecoderLayerPre_woSA(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(layer_num)])
        else:
            self.diff_decoders = nn.ModuleList([DecoderLayerPre_Enc(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(layer_num)])
        self.ln = nn.LayerNorm(self.module_dim, eps=1e-6)

    def forward(self, span, condition, time
                ):
        # 输入 span 噪声数据 b, d
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        time_emb = self.time_embed(time) # b, d
        # span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        # condition_time = condition + time_emb.unsqueeze(1)
        ### Span embedding
        output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb # b, d
        # output = self.noise_embed(span.to(torch.float32)) # b, d
        output = output.unsqueeze(1) # b, 1, d

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, condition)
        
        output = self.ln(output)

        return output.squeeze(1) # b, d

class DenoiseNetwork_sep(nn.Module):

    def __init__(self, module_dim=768, num_class=2):
        super().__init__()


        self.module_dim = module_dim
        self.num_class = num_class

        # 时间编码器
        # self.time_embed = SinusoidalPositionEmbeddings(d_model) # diff_mom 正余弦编码
        from DiT_module import TimestepEmbedder
        self.time_embed = TimestepEmbedder(self.module_dim) # DiT 正余弦加mlp
        
        # 噪声编码器
        # 4000 --> d
        # self.noise_embed = MLP(num_class,d_model, d_model, 1) # diff_mom MLP
        self.noise_embed = nn.Linear(self.num_class, self.module_dim) # DiT linear
        from transformer.Layers import DecoderLayer_woSA, DecoderLayerPre_woSA, DecoderLayer_Enc, DecoderLayer_context, DecoderLayer_multi
        self.diff_decoders = nn.ModuleList([DecoderLayer_multi(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(1)])

    def forward(self, span, condition, time
                ):
        # 输入 span 噪声数据 b, d
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        video, question = condition
        time_emb = self.time_embed(time) # b, d
        # span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        # condition_time = condition + time_emb.unsqueeze(1)
        ### Span embedding
        output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb # b, d
        # output = self.noise_embed(span.to(torch.float32)) # b, d
        output = output.unsqueeze(1) # b, 1, d

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, question,video )

        return output.squeeze(1) # b, d

class DenoiseNetwork_retrival(nn.Module):

    def __init__(self, module_dim=768, num_class=2):
        super().__init__()


        self.module_dim = module_dim
        self.num_class = num_class

        # 时间编码器
        # self.time_embed = SinusoidalPositionEmbeddings(d_model) # diff_mom 正余弦编码
        from DiT_module import TimestepEmbedder
        self.time_embed = TimestepEmbedder(self.module_dim) # DiT 正余弦加mlp
        
        # 噪声编码器
        # 4000 --> d
        # self.noise_embed = MLP(num_class,d_model, d_model, 1) # diff_mom MLP
        self.noise_embed = nn.Linear(self.num_class, self.module_dim) # DiT linear
        from transformer.Layers import DecoderLayer_woSA, DecoderLayerPre_woSA
        self.diff_decoders = nn.ModuleList([DecoderLayer(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(1)])

    def forward(self, span, condition, time
                ):
        # 输入 span 噪声数据 b, n
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        time_emb = self.time_embed(time) # b, d
        # span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        condition_time = condition + time_emb.unsqueeze(1)
        ### Span embedding
        # output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb.unsqueeze(1).repeat(1,span.shape[1],1) # b, n, d
        output = self.noise_embed(span.to(torch.float32)) # b, d
        output = output.unsqueeze(1) # b, 1, d

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, condition_time)

        return output.squeeze(1) # b, d

class DenoiseNetwork_DiT(nn.Module):

    def __init__(self, module_dim=768, num_class=2):
        super().__init__()


        self.module_dim = module_dim
        self.num_class = num_class

        # 时间编码器
        # self.time_embed = SinusoidalPositionEmbeddings(d_model) # diff_mom 正余弦编码
        from DiT_module import TimestepEmbedder
        self.time_embed = TimestepEmbedder(self.module_dim) # DiT 正余弦加mlp
        
        # 噪声编码器
        # 4000 --> d
        # self.noise_embed = MLP(num_class,d_model, d_model, 1) # diff_mom MLP
        self.noise_embed = nn.Linear(self.num_class, self.module_dim) # DiT linear
        from transformer.Layers import DecoderLayer_woSA, DecoderLayerPre_woSA, DecoderLayer_scale
        self.diff_decoders = nn.ModuleList([DecoderLayer_scale(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(1)])
        self.layer_norm = nn.LayerNorm(module_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(module_dim, 2 * module_dim, bias=True)
        )

        for block in self.diff_decoders:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    # def modulate(self, x, shift, scale):
    #     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, span, condition, time
                ):
        # 输入 span 噪声数据 b, n
        # 条件数据 b, l, d
        # 时间 即 强度 b
        ###  b, n   b, l, d    b
        ### 时间正余弦编码 b, d
        time_emb = self.time_embed(time) # b, d
        # span = inverse_sigmoid(span).sigmoid() # 反向sigmoid(sigmoid的逆函数)+sigmoid  # b, n
        # condition_time = condition + time_emb.unsqueeze(1)
        ### Span embedding
        # output = self.noise_embed(span.to(torch.float32)) + 0.1 * time_emb.unsqueeze(1).repeat(1,span.shape[1],1) # b, n, d
        output = self.noise_embed(span.to(torch.float32)) # b, d
        output = output.unsqueeze(1) # b, 1, d

        for diff_decoder in self.diff_decoders:
            output, _, _ = diff_decoder(output, condition, time_emb)

        shift, scale = self.adaLN_modulation(time_emb).chunk(2, dim=1)

        output = modulate(self.layer_norm(output), shift, scale)
        

        return output.squeeze(1) # b, d

class VQA(nn.Module):
    def __init__(self, **kwargs):
        super(VQA, self).__init__()
        self.app_pool5_dim = kwargs.pop("app_pool5_dim")
        self.motion_dim = kwargs.pop("motion_dim")
        self.num_frames = kwargs.pop("num_frames")
        self.word_dim = 768
        self.module_dim = 768
        self.question_type = kwargs.pop("question_type")
        self.num_answers = kwargs.pop("num_answers")
        self.is_lm_frozen = kwargs.pop("lm_frozen") ### 是否使用
        self.dropout = kwargs.pop("dropout")

        self.GCN_adj = kwargs.pop("GCN_adj")
        self.GAT_adj = kwargs.pop("GAT_adj")
        self.lm_name = kwargs.pop("lm_name")
        self.T = kwargs.pop("T")
        self.scale = kwargs.pop("scale")
        self.model_id = kwargs.pop("model_id")
        self.layer_num = kwargs.pop("layer_num")

        self.visual_dim = 768

        self.proj_v = nn.Linear(self.visual_dim, self.module_dim)
        # self.proj_t = nn.Linear(self.word_dim, self.module_dim)
        # self.proj_c = nn.Linear(768, self.module_dim)

        # self.v_cls = nn.Parameter(torch.randn(1, 1, self.module_dim))
        # self.q_cls = nn.Parameter(torch.randn(1, 1, self.module_dim))

        # self.v_pos = PositionalEncoding(self.module_dim)
        # self.t_pos = PositionalEncoding(self.module_dim)
        
        self.v_ln = nn.LayerNorm(self.module_dim, eps=1e-6)
        # self.t_ln = nn.LayerNorm(self.module_dim, eps=1e-6)
        from transformer.Layers import EncoderLayerPre, DecoderLayerPre, DecoderLayer_woSA
        # self.v_encoders = nn.ModuleList([EncoderLayer(self.module_dim, self.module_dim, 8, self.module_dim, self.module_dim) for i in range(1)])
        self.q_encoders = nn.ModuleList([EncoderLayer(self.module_dim, self.module_dim, 8, self.module_dim, self.module_dim) for i in range(1)]) 

        self.answer_decoders = nn.ModuleList([DecoderLayer(self.module_dim, self.module_dim, 1, self.module_dim, self.module_dim) for i in range(1)])



        self.classifier = nn.Sequential(
            nn.Linear(self.module_dim, self.module_dim),
            nn.ELU(), # ELU()
            nn.BatchNorm1d(self.module_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.module_dim, self.num_answers)
        )
        
        # from transformers import RobertaModel
        # self.lm = RobertaModel.from_pretrained('/root/autodl-tmp/roberta-large').cuda()
        # self.lm = AutoModel.from_pretrained('/root/autodl-tmp/roberta-base').cuda()
        self.lm = AutoModel.from_pretrained('/root/autodl-tmp/{}'.format(self.lm_name)).cuda()
        # from transformers import CLIPTextModel
        # self.lm = CLIPTextModel.from_pretrained('/root/autodl-tmp/clip-vit-large-patch14').cuda()
        if self.is_lm_frozen:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> frozen the lm >>>>>>>>>>>>>>>>>>>>>>>")
            for _, param in self.lm.named_parameters():
                param.requires_grad = False

        self.denoise_network = DenoiseNetwork(module_dim=self.module_dim, module_id=self.model_id, layer_num=self.layer_num, num_class=self.num_answers) 
        self.transformer =  Transformer(
            d_model=self.module_dim,
            dropout=self.dropout,
            num_queries=1,
            nhead=8,
            dim_feedforward=self.module_dim,
            num_encoder_layers=1,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=True,
            activation='prelu', # prelu
            num_class=self.num_answers,
            scale=self.scale,
            denoise_network=self.denoise_network,
            steps=self.T
        )
        
    # def init_weights(self):
    #     nn.init.kaiming_normal(self.proj_v.weight, mode='fan_out', nonlinearity='relu') # leaky_relu
    #     nn.init.constant()
        

    def forward(self, answer_token, video, question, ans_candidate, ans_candidate_len, gt_target=None):
        # import pdb; pdb.set_trace()
        question = self.lm(**question).last_hidden_state
        gt_answer = self.lm(**answer_token).last_hidden_state.mean(1) # b, d
        # question = self.ml.encode_text_hs(question).float() # 长度为 77
        # context = self.ml.encode_text_hs(context).mean(dim=1, keepdim=True).float()
        # app_pool5_feat, motion_feat: b, f, d
        # question: b, l, d 
        # context: b, 1, d (pooled)
        B = question.size(0)
        video_length = video.size(0)

        ### Encoder
        ## visual
        visual_feat = video
        # linear to model dim
        visual_feat = self.proj_v(visual_feat) # b, 2f, d
        # add learnable global token
        # visual_feat = torch.cat([self.v_cls.repeat(B, 1, 1), visual_feat], dim=1) # b, 1+2f, d
        # add postional embedding
        # visual_feat = self.v_pos(visual_feat)
        # # # normalization
        visual_feat = self.v_ln(visual_feat)
        # # encoder MHA + FFN
        # for v_encoder in self.v_encoders:
        #     visual_feat, _ = v_encoder(visual_feat)

        ## question
        # linear to model dim
        # question_feat = self.proj_t(question) # b, l, d
        question_feat = question
        # add learnable global token
        # question_feat = torch.cat([self.q_cls.repeat(B, 1, 1), question_feat], dim=1) # b, 1+l, d
        # add postional embedding
        # question_feat = self.t_pos(question_feat)
        # normalization
        # question_feat = self.t_ln(question_feat)
        # encoder MHA + FFN
        for q_encoder in self.q_encoders:
            question_feat, _ = q_encoder(question_feat)
        # question_enc = question_feat
        # question_raw = question_feat
        ### Answer Decoder
        # decoder q = q k=v = v local
        for answer_decoder in self.answer_decoders:
            question_feat, _, _ = answer_decoder(question_feat, visual_feat)
        mid_logits = self.classifier(question_feat.mean(1))
        # logits = torch.randn_like(mid_logits)
        # src = torch.cat([visual_feat, question_feat], dim=1)
        # question_feat = (question_raw, question_feat)
        # import pdb; pdb.set_trace()
        # question_feat = torch.cat([visual_feat, question_feat], dim=1)
        if gt_target is not None:
            gt_target = F.one_hot(gt_target, self.num_answers)
            logits = self.transformer(question_feat, gt_target) # b, d
        else:
            logits = self.transformer(question_feat, gt_target) # b, n
        # logits = self.classifier(question_feat.mean(1))
        # get q global
        # q_cls = question_feat[: , 0, : ]
        # classfier linear + ELU + linear 

        return logits, mid_logits
        # return logits
