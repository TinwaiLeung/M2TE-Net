import pandas as pd
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
 

class AVE(data.Dataset):
    ''' 
        10 sec. audio feature from logMel
        10 sec. audio feature from CLIP
    '''
    def __init__(self,data, CLIP_preprocess):
        super(AVE, self).__init__()
        self.classes_AVE = {
            'Accordion' : 0,
            'Acoustic guitar' : 1,
            'Baby cry, infant cry' : 2,
            'Banjo' : 3,
            'Bark' : 4,
            'Bus' : 5,
            'Cat' : 6,
            'Chainsaw' : 7,
            'Church bell' : 8,
            'Clock' : 9,
            'Female speech, woman speaking' : 10,
            'Fixed-wing aircraft, airplane' : 11,
            'Flute' : 12,
            'Frying (food)' : 13,
            'Goat' : 14,
            'Helicopter' : 15,
            'Horse' : 16,
            'Male speech, man speaking' : 17,
            'Mandolin' : 18,
            'Motorcycle' : 19,
            'Race car, auto racing' : 20,
            'Rodents, rats, mice' : 21,
            'Shofar' : 22,
            'Toilet flush' : 23,
            'Train horn' : 24,
            'Truck' : 25,
            'Ukulele' : 26,
            'Violin, fiddle' : 27
                }

        self.data = data 
        self.dataset_root = "E:/M2TE-Net/Dataset/AVE"
        self.fragments = []
        self.logMel_feature = Extract_logMel_feature()
        self.CLIP_preprocess = CLIP_preprocess

        f = open(os.path.join(self.dataset_root, self.data)+".txt", "r")
        lines = f.read().split('\n')
        for n, file in enumerate(lines):
            if file == '':
                continue
            if file.split('&')[1] == 'VideoID': 
                continue
            class_name = file.split('&')[0]
            file_name = file.split('&')[1]
            class_id = self.classes_AVE[class_name]
            self.fragments.append([ file_name, class_id ])
        
        self.length = len(self.fragments)

    def __len__(self):
        return  self.length
    
    @property
    def classes(self):
        classes_all = pd.read_csv('E:/M2TE-Net/Dataset/AVE/AVE_label.csv')
        return classes_all.values.tolist()
    
    def __getitem__(self,index):
        file_name, class_id  = self.fragments[index]

        audio_path = os.path.join(self.dataset_root, "audio", file_name + ".wav")
        x, _ = librosa.core.load(audio_path, sr=48000, mono=True)
        x_wav = torch.from_numpy(x)   
        x_wav = x_wav[: 480000]
        x_mel = self.logMel_feature(x_wav)
        x_mel = x_mel[:,:59]
        x_wav = x_wav[:479999]

        imgs = []  
        video_path = os.path.join(self.dataset_root, "keyframe", file_name)
        photo_names = os.listdir(video_path)
        photo_names = os_sorted(photo_names, reverse=False)                            
        for photo_name in photo_names[:19]: 
            img_add = os.path.join(video_path,photo_name)
            img = Image.open(img_add)
            img = self.CLIP_preprocess(img).unsqueeze(0)
            imgs.append(img)  
        imgs = torch.cat(imgs,dim=0)
        
        return x_wav, x_mel , imgs, class_id  
      
