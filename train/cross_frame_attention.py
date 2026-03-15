import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V):
        score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1))
        attn = F.softmax(score, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_bins=512, d_ff=512, d_k=64, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.feature_bins = feature_bins
        self.dk = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(feature_bins, d_k * n_heads)
        self.W_K = nn.Linear(feature_bins, d_k * n_heads)
        self.W_V = nn.Linear(feature_bins, d_k * n_heads)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.output = nn.Sequential(
            nn.Linear(n_heads * d_k, feature_bins),
            nn.Dropout(0.1),
        )
        self.layerNorm = nn.LayerNorm(d_k * n_heads)     
        

    def forward(self, Q, K, V):
        # Q, K, V: [batch_size x len_q x d_model]

        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        # v_s: [batch_size x n_heads x len_k x d_v]

        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,  self.n_heads * self.dk)
        output = self.output(context)
        residual, output = residual.view(batch_size, -1), output.view(batch_size, -1)
        result = self.layerNorm(output + residual)

        return result

class CFAM(nn.Module):
    def __init__(self,feature_bins=512, d_ff=512, d_k=64, n_heads=8):
        super(CFAM, self).__init__()
        self.cross_frame_attention = MultiHeadAttention(feature_bins=feature_bins, d_ff=d_ff, d_k=d_k, n_heads=n_heads)
        
    def forward(self,x):
        batch, L_seq, d_x = x.size(0), x.size(1), x.size(-1)
        x_list = torch.chunk(x, chunks = L_seq, dim = 1)  
        temp = [None]*L_seq
        output = [None]*L_seq
        for index, x_list_i in enumerate(x_list):
            temp[index] = x_list_i.unsqueeze(1).view(batch, 8, 64)
        for i in range(L_seq):
            output[i] = self.cross_frame_attention(temp[i], temp[0], temp[0])
        output = torch.stack(output).transpose(0,1).contiguous()   
        return output