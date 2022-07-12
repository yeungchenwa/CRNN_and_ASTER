from cgitb import text
from email.policy import strict
import os
import glob
import time
import yaml
import random
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from tools.yaml_config import Yaml
from models.builder import CRNN_builder
from tools.converter import CTCLabelConverter
from dataset.utils import get_data
from tools.utils import Averager

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/CRNN.yaml', help='path to config file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # get the args
    args = get_argparse()
    yaml_file = Yaml(args.config) 
    configs = yaml_file.read_yaml()
    model_config = configs['model']
    dataset_config = configs['dataset']
    train_config = configs['train']
    eval_config = configs['eval']

    # use cuda or not
    device = torch.device('cuda:0' if train_config['cuda'] else 'cpu')
    print(type(device))
    
    # cudnn
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    torch.manual_seed(configs['seed'])
    torch.cuda.manual_seed(configs['seed'])

    cudnn.benchmark = True
    cudnn.deterministic = True

    # build the converter
    if model_config['prediction'] == 'CTC':
        converter = CTCLabelConverter(dataset_config['character'])
    else:
        True

    # build the train dataset
    train_data_dir = [os.path.join(dataset_config['train_root'], dir_name) for dir_name in os.listdir(dataset_config['train_root'])]
    train_dataset, train_data_loader = get_data(train_data_dir, dataset_config=dataset_config, is_train=True, randomSampler=dataset_config['random_sampler'])
    print("Train dataset size is ", len(train_dataset))

    # build the test dataset
    test_data_dir = [os.path.join(dataset_config['test_root'], dir_name) for dir_name in os.listdir(dataset_config['test_root'])]
    test_dataset, test_data_loader = get_data(test_data_dir, dataset_config=dataset_config, is_train=False, randomSampler=dataset_config['random_sampler'])
    print("Test dataset size is ", len(test_dataset))

    # build the model
    if configs['name'] == 'CRNN':
        model = CRNN_builder(model_config['feat_channels'], model_config['hidden_size'], dataset_config['num_class'])
    else:
        True
    if train_config['checkpoint_path']:
        model.load_state_dict(torch.load(train_config['checkpoint_path']), strict=False)
        print("load model state dicet sucessfully!")
    model.to(device)
    model.train()

    # build the loss
    if model_config['prediction'] == 'CTC':
        criterion = nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg = Averager()
    
    # build thr optimizer
    if train_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=train_config['lr'])

    # train
    best_accuracy = 0
    for i in range(train_config['epoch']):
        for j, (imgs, labels, labels_length) in enumerate(train_data_loader):
            texts, length = converter.encode(labels)
            imgs = imgs.to(device)
            texts = texts.to(device)
            # print(texts.size())
            output = model(imgs)

            if 'CTC' in model_config['prediction']:
                # output_size = torch.IntTensor(output)
                input_length = torch.IntTensor([output.size(1)]*dataset_config['batch_size'])
                output = output.log_softmax(2).permute(1, 0, 2)
                # print(output.size())
                
                correct = 0
                _, out_max = output.max(2)
                out_max = out_max.transpose(1, 0).contiguous()
                out_text = converter.decode(out_max, input_length)
                for pred, target in zip(out_text, labels):
                    if pred == target:
                        correct += 1
                accuracy = correct / float(dataset_config['batch_size'])    
                
                # loss = criterion(output, texts, input_length, length) / dataset_config['batch_size']   # carefully
                loss = criterion(output, texts, input_length, length)
                print("Epoch {:d} Step {:d} loss is {:4f} accuracy is {:4f}".format((i+1), (j+1), loss, accuracy))
                
                loss_avg.add(loss)
            else:
                True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if j%100 == 0:       # save model per 1000 steps
                if configs['evaluate']:
                    n_correct = 0
                    for k, (imgs_t, labels_t, labels_length_t) in enumerate(test_data_loader):
                        # print(labels_t)
                        texts_t, length_t = converter.encode(labels_t, batch_max_length=dataset_config['batch_max_length'])
                        imgs_t = imgs_t.to(device)
                        texts_t = texts_t.to(device)

                        with torch.no_grad():
                            preds = model(imgs_t)

                            if 'CTC' in model_config['prediction']:
                                input_length_t = torch.IntTensor([preds.size(1)]*dataset_config['batch_size'])
                                preds = preds.log_softmax(2).permute(1, 0, 2)

                                # eval
                                _, out_max = preds.max(2)
                                out_max = out_max.transpose(1, 0).contiguous()
                                out_text = converter.decode(out_max, input_length_t)
                                # print(out_text)
                                
                                for pred, target in zip(out_text, labels_t):
                                    if pred == target:
                                        n_correct += 1
                                        print("test successfully, sample is {}".format(pred))
                            else:
                                True
                

                    accuracy = n_correct / float(len(test_dataset))
                    print('Test accuray: %f' % accuracy)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        print('Find the best parameter, saving the checkpoint...')
                        torch.save(model.state_dict(), '/home/yeung/DLVC/checkpoint/CRNN_and_Aster/best_accuracy.pth')
                        print('Saved Successfully!')
        print("Epoch {:d} total avg loss is {:4f}".format((i+1), loss_avg.out()))
        torch.save(model.state_dict(), '/home/yeung/DLVC/checkpoint/CRNN_and_Aster/epoch_{:d}.pth'%(i+1))