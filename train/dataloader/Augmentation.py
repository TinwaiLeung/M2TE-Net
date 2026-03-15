
from dataset.transforms_ss import *
# from autoaugment import RandAugment, ImageNetPolicy

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation(training):
    input_mean = [0.432, 0.413, 0.417]
    input_std = [0.231, 0.230, 0.231]

    scale_size = 224 * 256 // 224

    if training:

        unique = torchvision.transforms.Compose([
                                                 GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(False),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0),
                                                 Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean, input_std)]
                                                )
    else:
        unique = torchvision.transforms.Compose([
                                                 GroupScale(scale_size),
                                                 GroupCenterCrop(224),
                                                 Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean, input_std)])

    return unique

def randAugment(transform_train):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(2, 9)))
    return transform_train

def RandAugment(N, M):
    transforms = ['Identity', 'AutoContrast', 'Equalize', 
                  'Rotate', 'Solarize', 'Color', 'Posterize', 
                    'Contrast', 'Brightness', 'Sharpness', 
                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY']
    sampled_ops = np.random.choice(transforms, N)
    return [(op, M) for op in sampled_ops]