# M2TE-Net
Multimodal Multi-Scale Temporal Enhancement Network for Audio-Visual Scene Classification

# Usage
## Dataset preparation
For training on TAU, download data from **https://zenodo.org/records/4477542**. You will need to download the files TAU-urban-audio-visual-scenes-2021-development.audio.[1-8].zip and TAU-urban-audio-visual-scenes-2021-development.video.[1-16].zip. The directory should be organized as follows:

    TAU
    └───audio
    │   │  airport-helsinki-3-112.wav
    │   │  airport-helsinki-3-117.wav
    │   │  ...
    │   │  tram-vienna-285-8638.wav
    │   │  tram-vienna-285-8639.wav
    │   │  ...
    └───video
    |   └───airport
    │   │   │  airport-helsinki-3-112.mp4
    │   │   │  airport-helsinki-3-117.mp4
    │   │   │  ...
    |   └───bus
    │   │   │  bus-barcelona-15-599.mp4
    │   │   │  ...
    │   │  ...
    │   └───tram
    │   │   │  tram-barcelona-180-5558.mp4
    │   │   │  ...
    │   │   │  tram-vienna-285-8639.mp4

AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK. The directory should be organized as follows:

    AVE
    └───audio
    │   │  -_6ONBcE92A.wav
    │   │  -_bm7VDAUo4.wav
    │   │  ...
    │   │  -Zz2ZHMG6VA.wav
    │   │  -Zz2ZHMG6VA.wav
    │   │  ...
    └───video
    │   │  -_6ONBcE92A.mp4
    │   │  -_bm7VDAUo4.mp4
    │   │  ...
    │   │  -Zz2ZHMG6VA.mp4
    │   │  -Zz2ZHMG6VA.mp4
    │   │  ...

## Training
For the initialization of weights in the visual module, please download the pre-trained Resnet-50 model and CLIP/Resnet-50 model.

For training with the pre-trained Resnet-50 model，run：
python train_with_Resnet50.py

For training with the pre-trained CLIP/Resnet-50 model，run：
python train_with_CLIP.py
