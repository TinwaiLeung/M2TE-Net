import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss

import os
from tqdm import tqdm


from UMITENet_with_ResNet50 import Temporal_Attention   ## M2TE-Net 

from dataloader.dataloader_TAU_RN import TAU
from dataloader.AVE_Dataset_RN import AVE

''' train for model with ResNet50 '''
print("======================================")
date = 20250111
idx =2
m_gpu = 0
device = "cuda:{}".format(m_gpu)
print("Device: {}".format(device)) 
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % m_gpu
torch.cuda.set_device(m_gpu)
torch.cuda.is_available()
torch.cuda.current_device()

print("======================================")

np.random.seed(0)    
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False      

parser = argparse.ArgumentParser(description='Train audio, video and audio-visual networks')
parser.add_argument('--n_epoch', type=int, default=50,required=False,
                    help='number of epochs you wish to run') 
parser.add_argument('--batch_size', type=int, default=32,required=False,
                    help='set the batch size')        
args, _ = parser.parse_known_args()


def train():
    model.train()  
    train_loss = 0. 
    with tqdm(total=len(train_loader), desc='Train: ', leave=True, ncols=200, unit='batch', unit_scale=True, colour='CYAN') as pbar:
        for batch_idx, (x_wav, x_mel, imgs, class_id) in enumerate(train_loader):    
            x_wav, x_mel, imgs, class_id = x_wav.cuda(), x_mel.cuda(), imgs.cuda(), class_id.cuda()

            optimizer.zero_grad()
            predict = model(x_wav, x_mel, imgs)
            loss = loss_fn(predict, class_id)
            loss.backward()                      
            train_loss += loss.data.item()         
            optimizer.step()     

            pbar.set_postfix({'Train_loss ' : ' {:.4f}'.format(train_loss/(batch_idx+1))})
            pbar.update(1)
   
    # scheduler.step()
    train_loss /=  (batch_idx+1)
       
    return train_loss

def val():
    model.eval()
    y_pre = []
    y_true = []
    y_possible = []
    test_loss = 0.
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='val : ', leave=True, ncols=200, unit='batch', unit_scale=True, colour='YELLOW') as pbar:
            for batch_idx, (x_wav, x_mel, imgs, class_id) in enumerate(test_loader):
                x_wav, x_mel, imgs, class_id = x_wav.cuda(), x_mel.cuda(), imgs.cuda(), class_id.cuda()
                predict = model(x_wav, x_mel, imgs)
                loss = loss_fn(predict, class_id)
                
                possible = nn.functional.softmax(predict,dim=1) 

                _, pre = torch.max(predict,dim=1) 
                y_possible.append(possible.cpu())
                y_pre.append(pre.cpu())
                y_true.append(class_id.cpu())
                
                test_loss += loss.data.item()  
                pbar.set_postfix({'Test_loss  ' : ' {:.8f}'.format(test_loss/(batch_idx+1))})
                pbar.update(1)

    y_possible = torch.cat(y_possible).cpu().detach().numpy()
    y_pre = torch.cat(y_pre).cpu().detach().numpy() 
    y_true = torch.cat(y_true).cpu().detach().numpy() 

    logLoss = log_loss(y_true,y_possible)
    acc = accuracy_score(y_true, y_pre)
        
    return logLoss, acc     
   
if __name__ == "__main__":
    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime()) 
    csv_filename = 'epoch-{}_BatchSize-{}_date-{}_index-{}.csv'.format(args.n_epoch,args.batch_size,date,idx)
    df = pd.DataFrame(columns=['time', 'epoch', 'train_loss', 'val_loss', 'val_Acc'])
    df.to_csv(os.path.join('/code/TAU-urban-audio-visual-scenes/train/log_csv', csv_filename), index=False)

    print('-----------loading train_dataset')
    train_Dataset = AVE('train') 
    # train_Dataset = TAU('train', CLIP_preprocess) 
    train_loader = DataLoader(train_Dataset,batch_size = args.batch_size, 
                                    shuffle = True, num_workers = 16, drop_last = False)   
    print('-----------loading val_dataset')
    val_Dataset = AVE('val') 
    val_loader = DataLoader(val_Dataset,batch_size = args.batch_size, 
                                    shuffle = True, num_workers = 16, drop_last = False)  
    print('-----------loading test_dataset')
    test_Dataset = AVE('test')  # TAU_Dataset
    test_loader = DataLoader(test_Dataset,batch_size = args.batch_size,
                                    shuffle = True, num_workers = 16, drop_last = False)
    

    model = Temporal_Attention(num_classes = 10)
    model = model.cuda()

    output_dir = '/code/TAU-urban-audio-visual-scenes/train/model_audiovideo/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
        print("Directory " , output_dir ,  " Created ")
    else:
        print("Directory " , output_dir ,  " already exists")

    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer =optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 30], gamma=0.2)


    print('-----------start training')
    train_loss_list = []      
    val_loss_list = []
    val_Acc_list = [] 
    for epoch in range(1, 1+args.n_epoch):
        print(' ')
        print('Epoch: ', epoch)
        model = model.cuda()

        train_loss_epoch = train()
        train_loss_list.append(train_loss_epoch)

        val_loss_epoch, val_acc_epoch = val()
        val_loss_list.append(val_loss_epoch)
        val_Acc_list.append(val_acc_epoch)     
        
        if val_loss_list[-1] == np.min(val_loss_list):
            with open(output_dir+'AV_model_epoch-{}_BatchSize-{}_date-{}_index-{}.pt'.format(args.n_epoch,args.batch_size,date,idx), 'wb') as f:    # 保存为二进制文件
                torch.save(model.cpu().state_dict(), f) 
                print('Best model found and saved.')

        csv_time = time.strftime("%Yy%mm%dd-%H:%M:%S", time.localtime()) 
        csv_list = [csv_time, epoch, train_loss_epoch, val_loss_epoch, val_acc_epoch]
        csv_data = pd.DataFrame([csv_list])
        csv_data.to_csv(os.path.join('/code/TAU-urban-audio-visual-scenes/train/log_csv', csv_filename), mode='a', header=False, index=False)

        ####### plot the loss and val loss curve####
        minmum_val_index=np.argmin(val_loss_list)
        minmum_val=np.min(val_loss_list)
        plt.plot(train_loss_list,'r')
        plt.plot(val_loss_list,'b')
        plt.axvline(x=minmum_val_index,color='k',linestyle='--')
        plt.plot(minmum_val_index,minmum_val,'r*')

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.savefig('/code/TAU-urban-audio-visual-scenes/train/model_audiovideo/'+
                    'AV_loss_epoch-{}_BatchSize-{}_date-{}_index-{}.png'.format(args.n_epoch,args.batch_size,date,idx), dpi=300, bbox_inches='tight')  # 保存损失函数图
        plt.close()
