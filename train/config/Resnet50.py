import os

config = {}
data_train_opt = {}
config["data_dir"] = "/code/DCASE2021-T1B-main/dataset"
config["device_ids"] = [1]

data_train_opt['data'] = 20240122
data_train_opt['number'] = 1

data_train_opt['batch_size'] = 16
data_train_opt['epoch'] = 30
data_train_opt['split'] = 'train'
data_train_opt['lr'] = 0.0001
data_train_opt["decay_epoch"] = 20
data_train_opt["decay_rate"] = 0.5
data_train_opt["save_epoch"] = 1
data_train_opt["log_step"] = 50
data_train_opt["continue_model"] = ""

feat_training_file = '/code/DCASE2021-T1B-main/baseline_video/lr{}_batch{}_{}_{}'.format(data_train_opt['lr'],data_train_opt['batch_size'],data_train_opt['data'],data_train_opt['number'])
final_model_file = os.path.join(feat_training_file,"Final_model.pth")

data_train_opt["training_log"] = os.path.join(feat_training_file,"training_log.npy")
data_train_opt["txt"] = os.path.join(feat_training_file,"acc.txt")
if not os.path.exists(feat_training_file):
    os.makedirs(feat_training_file)
data_train_opt["best"] = os.path.join(feat_training_file,"acc_best.txt")


data_train_opt["feat_training_file"] = feat_training_file
data_train_opt["final_model_file"] = final_model_file
config["data_train_opt"] = data_train_opt


