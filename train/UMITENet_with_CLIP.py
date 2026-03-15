import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet18 import resnet18
from CrossModal_Interaction_Module_V2 import CIM

class AudioNet_ResNet18(nn.Module):
    def __init__(self,seq_Len=59, dim=512):
        super(AudioNet_ResNet18, self).__init__()
        self.AudioNet = resnet18()
        self.AudioNet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.AudioNet.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self.tgramnet = TgramNet(seq_Len=seq_Len, num_layer=3, mel_bins=dim, win_len=24576, hop_len=8208)
        self.fc = nn.Sequential(
                    nn.Conv1d(16, 10, kernel_size=3, stride=1, padding=1),      
                    nn.BatchNorm1d(10),     
                )
                
    def forward(self, x_wav, x_mel):
        x_mel = x_mel.unsqueeze(1)                  
        x_wav = x_wav.unsqueeze(1)               
        x_t = self.tgramnet(x_wav).unsqueeze(1)   
        x = torch.concat((x_mel, x_t), dim=1)    
        x = x.transpose(-1,-2)                    

        output = self.AudioNet(x)                  
        output = self.fc(output)                   
        return output 

class TgramNet(nn.Module):
    def __init__(self, seq_Len=59, num_layer=3, mel_bins=512, win_len=24576, hop_len=8208):
        super(TgramNet, self).__init__()
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.BatchNorm1d(seq_Len),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(seq_Len, seq_Len, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)      
        out = out.transpose(1,2)           
        out = self.conv_encoder(out)     
        out = out.transpose(1,2)           
        return out   
    

class Temporal_Attention(nn.Module):
    def __init__(self, num_classes = 28, dropout_rate=0.2):
        super(Temporal_Attention, self).__init__()
        self.audio_encoder = AudioNet_ResNet18(seq_Len=59, dim=512)
        self.video_pro = nn.Sequential(
            nn.Linear(512, 512),       
            # nn.Conv1d(20, 10, kernel_size=3, stride=1, padding=1),     ## TAU 
            nn.Conv1d(19, 10, kernel_size=3, stride=1, padding=1),     ## AVE
            nn.BatchNorm1d(10),  
            )
        
        self.CIM = CIM()

        self.LocalAttention_audio1 = LocalAttention1(embed_dim=512, num_heads=8, interval=1)
        self.MTCM_audio = Multiscale_TemporalCorrelation(in_channels=512, dilation_rates=[1, 2, 5])

        self.classifier = nn.Sequential(           
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_wav, x_mel, imgs):
        '''
        x_wav: [64, 480000]
        x_mel: [64, 512, 59]audio
        imgs:  [64, 20,  51]
        '''

        A_feature = self.audio_encoder(x_wav, x_mel)   
        V_feature = self.video_pro(imgs)                

        A_Attn_feature = self.MTCM_audio(A_feature)  ##
        V_Attn_feature = self.MTCM_audio(V_feature)

        A_Attn_feature = self.LocalAttention_audio1(A_Attn_feature)  ##
        V_Attn_feature = self.LocalAttention_audio1(V_Attn_feature)

        mixed = self.CIM(V_Attn_feature,A_Attn_feature)
        result = self.classifier(mixed)    
        result = torch.mean(result,dim=1)+torch.max(result,dim=1)[0] 

        return result

class LocalAttention1(nn.Module):         
    def __init__(self, embed_dim, num_heads=4, interval=3):
        super(LocalAttention1, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.interval = interval
 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layerNorm = nn.LayerNorm(embed_dim)
 
    def forward(self, x):
        residual = x
        B, T, D = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
 
        # Create local attention masks
        masks = []
        for b in range(B):
            mask = torch.zeros((self.num_heads, T, T), device=x.device)
            for t in range(T):
                left = t - self.interval
                if left >= 0:
                    mask[:, t, left] = 1

                middle = t
                mask[:, t, middle] = 1

                right = t + self.interval
                if right < T:
                    mask[:, t, right] = 1
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # [B, num_heads, T, T]
 
        # Apply masks to attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, T, T]
        scores = scores.masked_fill(masks == 0, float('-inf'))  # Apply masks
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, T, T]
 
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
 
        # Final linear layer
        output = self.out_proj(attn_output)  # [B, T, D]

        residual_ouput = self.layerNorm(output + residual)
        return residual_ouput


class Multiscale_TemporalCorrelation(nn.Module):       
    def __init__(self, in_channels=512, dilation_rates=[1, 2, 5]):
        super(Multiscale_TemporalCorrelation, self).__init__()

        ## Function for short term information
        self.dilation_rates = dilation_rates
        self.MultiScaleDilatedConv = nn.ModuleList([
                nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 4, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(in_channels // 4),
                nn.ReLU()) for dilation in dilation_rates
        ])

        ## Function for long term information
        self.conv = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0)
        self.q_conv = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=1, bias=False)
        self.scale_factor = nn.Parameter(torch.tensor((in_channels // 4) ** 0.5), requires_grad=False)
        self.out_conv = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=1)

        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

 
    def forward(self, x):
        B, T, D = x.size()  # [B, T, D]

        Shortterm_information = []
        for conv in self.MultiScaleDilatedConv:
            output = conv(x.transpose(1, 2)).transpose(1, 2)          
            Shortterm_information.append(output)                         
        Shortterm_information = torch.cat(Shortterm_information, dim=-1)     
        redisual = self.conv(x.transpose(1, 2)).transpose(1, 2)                
        Q = self.q_conv(redisual.transpose(1, 2)).transpose(1, 2)              
        K = self.k_conv(redisual.transpose(1, 2)).transpose(1, 2)         
        V = self.v_conv(redisual.transpose(1, 2)).transpose(1, 2)                 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor     
        attn_weights = F.softmax(scores, dim=-1)                                
        Longterm_information = torch.matmul(attn_weights, V)                                    
        Longterm_information = self.out_conv(Longterm_information.transpose(1, 2)).transpose(1, 2)
        Longterm_information = Longterm_information + redisual

        fusion_information = torch.cat([Shortterm_information, Longterm_information], dim=-1)     
        fusion_information = self.fusion_conv(fusion_information.transpose(1, 2)).transpose(1, 2)  

        output = fusion_information + x
        return output

class LocalAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads=4, interval1=1, interval2=2):
        super(LocalAttention2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.interval1 = interval1
        self.interval2 = interval2
 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layerNorm = nn.LayerNorm(embed_dim)
 
    def forward(self, x):
        residual = x
        B, T, D = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
 
        # Create local attention masks
        masks1 = []
        for b in range(B):
            mask1 = torch.zeros((self.num_heads, T, T), device=x.device)
            for t in range(T):
                left = t - self.interval1
                if left >= 0:
                    mask1[:, t, left] = 1

                middle = t
                mask1[:, t, middle] = 1

                right = t + self.interval1
                if right < T:
                    mask1[:, t, right] = 1
            masks1.append(mask1)
        masks1 = torch.stack(masks1, dim=0)  # [B, num_heads, T, T]

        masks2 = []
        for b in range(B):
            mask2 = torch.zeros((self.num_heads, T, T), device=x.device)
            for t in range(T):
                left = t - self.interval2
                if left >= 0:
                    mask2[:, t, left] = 1

                middle = t
                mask2[:, t, middle] = 1

                right = t + self.interval2
                if right < T:
                    mask2[:, t, right] = 1
            masks2.append(mask2)
        masks2 = torch.stack(masks2, dim=0)  # [B, num_heads, T, T]
 
        # Apply masks to attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, T, T]
        scores1 = scores.masked_fill(masks1 == 0, float('-inf'))  # Apply masks
        scores2 = scores.masked_fill(masks2 == 0, float('-inf'))  # Apply masks
        attn_weights1 = F.softmax(scores1, dim=-1)  # [B, num_heads, T, T]
        attn_weights2 = F.softmax(scores2, dim=-1)  # [B, num_heads, T, T]
 
        # Compute attention output
        attn_output1 = torch.matmul(attn_weights1, v)  # [B, num_heads, T, head_dim]
        attn_output2 = torch.matmul(attn_weights2, v)  # [B, num_heads, T, head_dim]
        attn_output1 = attn_output1.transpose(1, 2).contiguous().view(B, T, D)  
        attn_output2 = attn_output2.transpose(1, 2).contiguous().view(B, T, D)  
 
        # Final linear layer
        output1 = self.out_proj(attn_output1)  
        output2 = self.out_proj(attn_output2) 

        residual_output1 = self.layerNorm(output1 + residual) 
        residual_output2 = self.layerNorm(output2 + residual) 
        output = residual_output1 + residual_output2 
        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
 
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layerNorm = nn.LayerNorm(embed_dim)
 
    def forward(self, x):
        residual = x
        B, T, D = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, T, head_dim]
 
        # Apply masks to attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, T, T]
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, T, T]
 
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
 
        # Final linear layer
        output = self.out_proj(attn_output)  # [B, T, D]

        residual_ouput = self.layerNorm(output + residual)
        return residual_ouput
