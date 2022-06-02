import os
import glob
import time
import random
import argparse

import torch.nn as nn
import torchvision.models as models
from yaml import parse
from models.backbone import resnet

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=float, default='./configs/CRNN.yaml', help='path to config file')
    args = parser.parse_args()
    return args

def train():
    return