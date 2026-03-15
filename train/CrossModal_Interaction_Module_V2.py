import torch
import torch.nn as nn
import Global_Interaction_Module
import sub_attention_mixed
from MCRN import MCRN
from MCAM_V2 import MCAM
import torch.nn.functional as F


''' 
    input:  [64, 10, 512]
    output: [64, 10, 512]
'''

class ForwardBackwardFusionModule(nn.Module):
    def __init__(self, d_model=512, d_k=64, n_head=8):
        super(ForwardBackwardFusionModule, self).__init__()
        self.dropout = nn.Dropout(0.1)
        sub_attention_mixed.init_gobal_variable(a=d_model,b=d_model,c=d_k,d=n_head)   
        self.cross_attention = sub_attention_mixed.Encoder(1)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        global_feat = video_feat * audio_feat                             
        memory = torch.cat([audio_feat, video_feat], dim=0)                
        mid_out = self.cross_attention(global_feat, memory)[0]    
        output = self.norm1(global_feat + self.dropout(mid_out))           

        return  output  # 输入输出维度相同

class conv_block_1(nn.Module):
    def __init__(self, dim = 512):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

    def forward(self, x):
        x = self.conv(x)          
        return x

class conv_block_2(nn.Module):
    def __init__(self, dim = 512):
        super(conv_block_2, self).__init__()
        self.conv = nn.Sequential(
                nn.Linear(dim, dim*2),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

    def forward(self, x):
        x = self.conv(x)           
        return x

class CIM(nn.Module):
    def __init__(self):
        super(CIM, self).__init__()

        self.MCAM1 = MCAM(feature_bins=512, d_ff=512, d_k=64, n_heads=8)
        self.MCAM2 = MCAM(feature_bins=256, d_ff=512, d_k=64, n_heads=8)
        self.MCAM3 = MCAM(feature_bins=128, d_ff=512, d_k=64, n_heads=8)
        self.MCAM4 = MCAM(feature_bins=64,  d_ff=512, d_k=64, n_heads=8)

        self.conv1 = conv_block_1(512)  
        self.conv2 = conv_block_1(256)
        self.conv3 = conv_block_1(128)
        self.conv4 = conv_block_2(64)

        self.classifier = nn.Linear(1024, 512)
        
    def forward(self,imgs, audios):
        mixed_1 = self.MCAM1(audios,imgs)                  

        imgs, audios = self.conv1(imgs), self.conv1(audios)  
        mixed_2 = self.MCAM2(audios,imgs)                    
        
        imgs, audios = self.conv2(imgs), self.conv2(audios)  
        mixed_3 = self.MCAM3(audios,imgs)                    

        imgs, audios = self.conv3(imgs), self.conv3(audios) 
        mixed_4 = self.MCAM4(audios,imgs)                    

        mixed_4 = self.conv4(mixed_4)                       
        output = torch.concat((mixed_4,mixed_3), dim=-1)     
        output = torch.concat((output,mixed_2), dim=-1)      
        output = torch.concat((output,mixed_1), dim=-1)      

        output = self.classifier(output)                  

        return output
