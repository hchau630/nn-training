import argparse
import pathlib

import torch
import torchvision.datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

from nn_training import config, transforms, datasets

def model_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-a', '--arch', type=str)
    parser.add_argument('-p', '--pretrained', action='store_true')
    
    return parser

def get_model(*, arch='resnet50', pretrained=False):
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
    else:
        print("=> creating model '{}'".format(arch))
        
    model = models.__dict__[arch](pretrained=pretrained)
    
    return model

def criterion_parser():
    parser = argparse.ArgumentParser()
    
    return parser
        
class Criterion(torch.nn.CrossEntropyLoss):
    def forward(self, input, target):
        loss = super().forward(input, target['label'])
        info = {}
        return loss, info
    
def get_criterion():
    criterion = Criterion()
    
    return criterion

def optimizer_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-l', '--lr', '--learning-rate', dest='lr', type=float)
    parser.add_argument('-m', '--momentum', type=float)
    parser.add_argument('-w', '--wd', '--weight-decay', dest='weight_decay', type=float)
    
    return parser

def get_optimizer(parameters, *, lr=0.1, momentum=0.9, weight_decay=1.0e-4):
    optimizer = torch.optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

def scheduler_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-s', '--step-size', type=float)
    parser.add_argument('-g', '--gamma', type=float)
    
    return parser

def get_scheduler(optimizer, *, step_size=30, gamma=0.1):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return scheduler

def dataset_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-c', '--crop-size', type=int)
    
    return parser

def get_dataset(*, crop_size=224):
    train_dataset = datasets.RandomTransformDataset(
        torchvision.datasets.ImageNet(config['paths']['imagenet'], split='train'),
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    )

    return train_dataset