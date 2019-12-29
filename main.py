import sys
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import torch.nn.fucntional as F 
import numpy as np
import yaml
from scipy import stats
from tensorboardX import SummaryWriter
from datetime import datetime
from torch import optim
import model 
import IQAdataset as Dataset

str_cd = 'exp/BiNet/' + datetime.now().strftime("%Y%m%d_%H%M%S")
if os.path.exists(str_cd) == False:
    os.makedirs(str_cd)

def get_data_loader (args,config):
    dataset = args.dataset
    Dataset = Dataset.dataset
    train_dataset = Dataset(config,'train')
    train_loader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle = True,
                                  num_workers = 4)
    val_dataset = Dataset(config,'test')
    val_loader = DataLoader(val_dataset)
    return train_loader,val_loader

def create_summary_writer(model, data_loader, log_dir = 'tensorboard_logs'):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def run(args,config,log_dir):
    model = model.args.model(eature_channels=config['feature_channels'], n1_nodes=config['n1_nodes'], n2_nodes=config['n2_nodes'])
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    train_loader,val_loader = get_data_loader(args,config)

    writer = create_summary_writer(model,train_loader,args.log_dir)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr = args.lr, weight_decay = args.weight_decay)


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
    parser.add_argument('--log_dir',type=str,default=str_cd +'/tensorboard_logs',help='log diretory for Tensorboard log output')


    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    with open(str_cd+'/config.yaml', "w")  as f:
        yaml_obj = {}
        yaml_obj['dataset'] = args.dataset
        yaml_obj['model'] = args.model
        yaml_obj['lr'] = args.lr 
        yaml_obj['batch_size'] = args.batch_size
        yaml_obj['epochs'] = args.ephoes
        yaml.dump(yaml_obj,f)
    config.update(config[args.dataset])
    config.update(config[args.model])

    if not os.path.exists(str_cd+'/results'):
        os.makedirs(str_cd+'/results')
    
    
    run(args = args,config = config )

                                