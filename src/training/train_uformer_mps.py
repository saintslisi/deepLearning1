import os
import sys

# --- GESTIONE PATH ---
# ---------------------

import argparse
from src.training import options
from src import utils
try:
    from src.datasets.uformer_denoise_dataset import get_training_data, get_validation_data
except ImportError:
    from src.datasets.uformer_denoise_dataset import get_training_data, get_validation_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from src.training.losses import CharbonnierLoss
from tqdm import tqdm 
import torch.nn.functional as F

# --- CLASSE PER IL LOGGING SU FILE ---
class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

######### Parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()

# --- PARAMETRI UFORMER-B (Hardcoded per sicurezza) ---
opt.depths = [1, 2, 8, 8]
opt.embed_dim = 32
opt.win_size = 8
opt.token_projection = 'linear'
opt.token_mlp = 'leff'

######### Set Device (MAC MPS) ###########
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS Acceleration")
else:
    device = torch.device("cpu")
    print("MPS not found, using CPU (Slow)")

######### Logs dir & Logger Setup ###########
log_dir = os.path.join(opt.save_dir, 'denoising', opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Nome del file log con data e ora
logname = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.txt')

# ATTIVAZIONE LOGGER: Da qui in poi ogni print finisce anche nel file
sys.stdout = Logger(logname)

print("Now time is : ", datetime.datetime.now().isoformat())
print(f"Log saved to: {logname}")

result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

######### Set Seeds ###########
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

######### Model ###########
print(f"===> Creating Model: {opt.arch}")
model_restoration = utils.get_arch(opt)

# --- CARICAMENTO PESI PRE-TRAINED ---
if opt.pretrain_weights and os.path.exists(opt.pretrain_weights):
    print(f"===> Loading Pretrained Weights from: {opt.pretrain_weights}")
    checkpoint = torch.load(opt.pretrain_weights, map_location='cpu')
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
        
    model_restoration.load_state_dict(new_state_dict, strict=False)
else:
    print("No Pretrained weights loaded/found! Training from scratch?")

model_restoration.to(device)

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### Scheduler DISABILITATO (Constant LR) ###########
# Abbiamo rimosso StepLR e Warmup per mantenere il LR fisso.
print(f"Using Constant Learning Rate: {opt.lr_initial}")

######### Resume (Training Interrotto) ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    
    # --- FIX LR RESUME ---
    # Non carichiamo il LR dal file (che potrebbe essere bassissimo),
    # ma forziamo quello specificato nei parametri (es. 1e-5 o 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr_initial
        
    print(f"===> Resuming Training. FORCING Learning Rate to: {opt.lr_initial}")

######### Loss ###########
criterion = CharbonnierLoss().to(device)

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=0, pin_memory=False, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### Validation Initiale ###########
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []

    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].to(device)
        input_ = data_val[1].to(device)

        # Center Crop 512x512
        _, _, h, w = input_.size()
        crop_size = 512
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        input_crop = input_[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
        target_crop = target[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        restored = model_restoration(input_crop)
        restored = torch.clamp(restored,0,1)  
        
        psnr_dataset.append(utils.batch_PSNR(input_crop, target_crop, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target_crop, False).item())

    if len_valset > 0:
        psnr_dataset = sum(psnr_dataset)/len_valset
        psnr_model_init = sum(psnr_model_init)/len_valset
        print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))

######### Training Loop ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) # Valida una volta per epoca

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0): 
        optimizer.zero_grad()

        target = data[0].to(device)
        input_ = data[1].to(device)

        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
            
        restored = model_restoration(input_)
        loss = criterion(restored, target)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        #### Evaluation ####
        if (i+1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []

                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].to(device)
                    input_ = data_val[1].to(device)
                    
                    # Center Crop 512x512
                    _, _, h, w = input_.size()
                    crop_size = 512
                    start_h = (h - crop_size) // 2
                    start_w = (w - crop_size) // 2
                    input_crop = input_[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
                    target_crop = target[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
                    
                    restored = model_restoration(input_crop)
                    restored = torch.clamp(restored,0,1)  
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target_crop, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR: %.4f\t] ----  [Best_PSNR %.4f] " % (epoch, i, psnr_val_rgb, best_psnr))
                model_restoration.train()
    
    # NOTA: Abbiamo rimosso scheduler.step() per mantenere il LR costante
    
    print("------------------------------------------------------------------")
    # Stampiamo il LR corrente dall'optimizer per conferma
    current_lr = optimizer.param_groups[0]['lr']
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time,epoch_loss, current_lr))
    print("------------------------------------------------------------------")
    
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 

print("Training Finished. Time: ",datetime.datetime.now().isoformat())