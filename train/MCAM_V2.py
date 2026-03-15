import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import sub_attention

class linear_QKV(nn.Module):
    def __init__(self, feature_bins=512, d_k=64, n_heads=8):
        super(linear_QKV, self).__init__()
        self.dk = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(feature_bins, d_k * n_heads)
        self.W_K = nn.Linear(feature_bins, d_k * n_heads)
        self.W_V = nn.Linear(feature_bins, d_k * n_heads)

    def forward(self, feature):
        batch_size = feature.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(feature).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        k_s = self.W_K(feature).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        v_s = self.W_V(feature).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        # v_s: [batch_size x n_heads x len_k x d_v]

        return q_s, k_s, v_s

class MCAM(nn.Module):
    def __init__(self, feature_bins=512, d_ff=512, d_k=64, n_heads=8):
        super(MCAM, self).__init__()
        self.dk = d_k
        self.n_heads = n_heads
        self.audio_QKV = linear_QKV(feature_bins=feature_bins, d_k=d_k, n_heads=n_heads)
        self.video_QKV = linear_QKV(feature_bins=feature_bins, d_k=d_k, n_heads=n_heads)

        self.output = nn.Linear(n_heads * d_k, feature_bins)
        self.layerNorm = nn.LayerNorm(feature_bins)

        sub_attention.init_gobal_variable(a=feature_bins, b=d_ff, c=d_k, d=n_heads)
        self.audio_self_attention = sub_attention.Encoder(1)   
        sub_attention.init_gobal_variable(a=feature_bins, b=d_ff, c=d_k, d=n_heads)
        self.video_self_attention = sub_attention.Encoder(1)   

    def forward(self, audio_feat, video_feat):
        audio_self,_ = self.audio_self_attention(audio_feat) 
        video_self,_ = self.video_self_attention(video_feat)

        batch_size = audio_feat.size(0)
        Q_audio, K_audio, V_audio = self.audio_QKV(audio_feat)
        Q_video, K_video, V_video = self.video_QKV(video_feat)
        score_audio = torch.matmul(Q_audio, K_audio.transpose(-1, -2)) / np.sqrt(K_audio.size(-1))
        attn_audio = F.softmax(score_audio, dim=-1)
        score_video = torch.matmul(Q_video, K_video.transpose(-1, -2)) / np.sqrt(K_video.size(-1))
        attn_video = F.softmax(score_video, dim=-1)
        attn = torch.einsum('ijkl,ijkl->ijkl', [attn_audio, attn_video])

        context_audio = torch.matmul(attn, V_audio)
        context_video = torch.matmul(attn, V_video)
        
        context = context_audio * context_video
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,  self.n_heads * self.dk)
        context = self.output(context)

        output = self.layerNorm(audio_feat + context + audio_self + video_self) 
        
        return output
