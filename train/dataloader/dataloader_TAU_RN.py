import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
import numpy as np
import os
import h5py
import csv
import torchaudio
import librosa
from torchvision import transforms
from natsort import os_sorted
import pandas as pd

class Extract_logMel_feature(object):
    def __init__(self, sr=48000,    # 1000ms
                 n_fft=24576,       # 512 ms, 和 win_length 相同
                 n_mels=512,
                 win_length=24576,  # 512 ms
                 hop_length=8208,   # 171 ms
                 power=2.0
                 ):
        super(Extract_logMel_feature, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        output = self.amplitude_to_db(self.mel_transform(x))
        return output      

class TAU(data.Dataset):
    def __init__(self,data):
        super(TAU, self).__init__()
        self.classes_dic = {'airport': 0,
                       'bus': 1,
                       'metro': 2,
                       'metro_station': 3,
                       'park': 4,
                       'public_square': 5,
                       'shopping_mall': 6,
                       'street_pedestrian': 7,
                       'street_traffic': 8,
                       'tram': 9}
        self.id2class = ['airport',
                       'bus',
                       'metro',
                       'metro_station',
                       'park',
                       'public_square',
                       'shopping_mall',
                       'street_pedestrian',
                       'street_traffic',
                       'tram']
        self.data = data 
        self.audio_root = "E:/M2TE-Net/Dataset/TAU/audio"
        self.dataset_root = "E:/M2TE-Net/Dataset/TAU"
        self.fragments = []
        self.logMel_feature = Extract_logMel_feature()
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.432,0.413,0.417],
                          std=[0.231, 0.230, 0.231])
            ] )

        if self.data == 'test':
            with open(os.path.join(self.dataset_root, self.data)+".csv", 'r') as f:
                reader = csv.reader(f)
                x = 0
                for i in reader:
                    if x ==0:
                        x = 1
                        continue
                    fragment_id = os.path.splitext(i[0])[0]  
                    class_id = fragment_id.split('-')[0]      
                    self.fragments.append([ fragment_id, self.classes_dic[class_id] ])
        else:
            with open(os.path.join(self.dataset_root, self.data)+".csv", 'r') as f:
                reader = csv.reader(f)
                x = 0
                for i in reader:
                    if x ==0:
                        x = 1
                        continue
                    file_name = os.path.split(i[1])[-1]           
                    fragment_id = os.path.splitext(file_name)[0]  
                    class_id = i[2]        
                    self.fragments.append([ fragment_id, self.classes_dic[class_id] ])

        self.length = len(self.fragments)

    def __len__(self):
        return  self.length
    
    @property
    def classes(self):
        classes_all = pd.read_csv('E:/M2TE-Net/Dataset/TAU/label.csv')
        return classes_all.values.tolist()
    
    def __getitem__(self,index):
        fragment_id, class_id  = self.fragments[index]

        audio_path = os.path.join(self.audio_root, fragment_id + ".wav")
        x, _ = librosa.core.load(audio_path, sr=48000, mono=True)
        x_wav = torch.from_numpy(x)  
        x_wav = x_wav[: 480000]
        x_mel = self.logMel_feature(x_wav)
        x_mel = x_mel[:,:59]
        x_wav = x_wav[:479999]


        imgs = []  
        video_path = os.path.join(self.dataset_root, "video frame", self.id2class[class_id], fragment_id)
        photo_names = os.listdir(video_path)
        photo_names = os_sorted(photo_names, reverse=False)                            
        for photo_name in photo_names:              
            img_add = os.path.join(video_path,photo_name)
            img = Image.open(img_add)
            img = img.resize((224, 224))
            img = img.convert("RGB")
            img = self.transforms(img)
            imgs.append(img.unsqueeze(0))      

        imgs = torch.cat(imgs,dim=0)          
        
        return x_wav, x_mel , imgs, class_id            
      
