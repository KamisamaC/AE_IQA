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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric

class IQAPerformance(Metric):
    
    def  __init__(self):
        self.y_pred = []
        self.y = []

    def update(self,output):
        pred , y = output

        self.y.append(y[0])
        self.y_pred.append(torch.mean(pred[0]))
    
    def conpute(self):
        y = np.reshape(np.asarray(self.y),(-1,))
        pred = np.reshape(np.asarray(self.y_pred)(-1,))

        srocc = stats.spearmanr(y,pred)[0]
        krocc = stats.stats.kendalltau(y,pred)[0]
        plcc = stats.pearsonr(y,pred)[0]
        rmse = np.sqrt(((y - pred) ** 2).mean())

        return srocc,krocc,plcc,rmse


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

        global best_criterion   
    best_criterion = -1  # SROCC>=-1
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

##################################################################
#Save the best result in validation dataset 
#Write information into TensorboardX
##################################################################
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        iter = engine.state.iteration % len(train_loader)
        if iter == len(train_loader)-1:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output), flush=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def change_lr(engine):
        for p in optimizer.param_groups:        
            print('lr: ', str(p['lr']))
        if engine.state.epoch % 25== 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print("Validation Results - Epoch: {}  SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            torch.save(model.state_dict(), trained_model_file)
###############################################################


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            #Acc = metrics['IDC_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)
            #3writer.add_scalar("testing/Acc", Acc, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"] > 0:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            #Acc = metrics['IDC_performance']
            global best_epoch
            print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                .format(best_epoch,  SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            np.save(save_result_file, ( SROCC, KROCC, PLCC, RMSE, MAE, OR))

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


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

                                