#performs the testing on all occlusions 

from __future__ import print_function
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import rm_occ_dataset
from datasets.ava_dataset import Ava 
from core.rm_optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss, RegionLoss_Ava
from core.model import YOWO, get_fine_tuning_parameters
from pathlib import Path
import glob 
####### Load configuration arguments
# ---------------------------------------------------------------
args  = parser.parse_args()
cfg   = parser.load_config(args)


####### Check backup directory, create if necessary
# ---------------------------------------------------------------
if not os.path.exists(cfg.BACKUP_DIR):
    os.makedirs(cfg.BACKUP_DIR)


####### Create model
# ---------------------------------------------------------------
model = YOWO(cfg)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
# print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

seed = int(time.time())
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)


####### Create optimizer
# ---------------------------------------------------------------
parameters = get_fine_tuning_parameters(model, cfg)
optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
best_score   = 0 # initialize best score
# optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/batch_size, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


####### Load resume path if necessary
# ---------------------------------------------------------------
if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
    # best_score = checkpoint['score']
    model.load_state_dict(checkpoint['state_dict'])
    print("checkpoint optmizer", checkpoint['optimizer'].keys())
    #dont load the optimizer
    # optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    # print("Loaded model score: ", checkpoint['score'])
    print("===================================================================")
    del checkpoint


####### Create backup directory if necessary
# ---------------------------------------------------------------
if not os.path.exists(cfg.BACKUP_DIR):
    os.mkdir(cfg.BACKUP_DIR)


####### Data loader, training scheme and loss function are different for AVA and UCF24/JHMDB21 datasets
# ---------------------------------------------------------------
dataset = cfg.TRAIN.DATASET
print("datset is--------->",cfg.TRAIN.DATASET)
assert dataset == 'ucf24' or dataset == 'jhmdb21' or dataset == 'ava', 'invalid dataset'



if dataset=='ucf24':
    occ_conf_path = Path('./occ_configs')/'ucf'/'**/*.yaml'
else:
    occ_conf_path = Path('./occ_configs')/'jhmdb'/'**/*.yaml'

yaml_paths = sorted(glob.glob(str(occ_conf_path),recursive =True))
# data_root = cfg.TRAIN.OCCLUSION_DATA_DIR+ '/**/*.yaml'
data_root = cfg.TRAIN.OCCLUSION_DATA_DIR

print("yaml paths", yaml_paths)
# exit(1)
if dataset == 'ava':
    pass

elif dataset in ['ucf24', 'jhmdb21']:

    for yaml_path in yaml_paths:
        
        
        test_dataset  = rm_occ_dataset.UCF_JHMDB_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, 
                                    yaml_path,data_root, dataset=dataset,
                        shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)

        
        test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)


        test  = getattr(sys.modules[__name__], 'test_ucf24_jhmdb21')

        test(cfg,yaml_path, 0, model, test_loader)

####### Training and Testing Schedule
# ---------------------------------------------------------------
print("rajat performing testing")
