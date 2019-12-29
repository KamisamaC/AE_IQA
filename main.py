import sys
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import torch.nn.fucntional as F 
import numpy as np
from scipy import stats
from tensorboardX import SummaryWriter

import model 
from IQAdataset import Dataset

def get_data_loader (arg,config,batch_size):
    train_dataset = Dataset.arg.dataset(config,'train')
    train_loader = DataLoader(train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 4)
    val_dataset = Dataset.arg.dataset(config,'test')
    val_loader = DataLoader(val_dataset)
    return train_loader,val_loader

def create_summary_writer(model, data_loader, log_dir = parser.log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def run()

if __name__ == "__main":
    parser = ArgumentParser(description = "Attention-Expaned CNNIQA")
    parser.add_argument('--batch_size',type = int , default=32,help = 'input batchsize for training(default:32)')
    parser.add_argument('--epochs',type=int,default=100,help = 'number of ephoes to train (default:100)')
    parser.add_argument('--lr',type=float,default=10e-4,help='learning rate(default:10e-4)')
    parser.add_argument('--weight_decay',type=float,default=10e-4,help='Weight_decay(default:10e-4)')
    parser.add_argument('--config',type=str,default='./config.yaml;',help='config file path(default:config.yaml)')
    parser.add_argument('--dataset',type=str,default='LIVE',help='dataset to train/test(default:LIVE2016)')
    parser.add_argument('--model',type=str,default='ResNet',help='Model to train/test(default:ResNet)')
    parser.add_argument('--use_gpu',type=bool,default=True,help='flag whether to use GPU acceleration')
    parser.add_argument('--log_dir',type=str,default='./tensorboard_logs',help='log diretory for Tensorboard log output')
    


                                