import argparse
import pathlib

import torch
import torchvision.datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

from nn_training import config, transforms, datasets, pca, distributed
import utils

def model_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-q', '--queue-size', type=int)
    parser.add_argument('-t', '--threshold', type=float)
    
    return parser

class Model(torch.nn.Module):
    def __init__(self, queue_size, threshold):
        super().__init__()
        
        resnet50 = models.resnet50(pretrained=False)
        self.fc = resnet50.fc
        self.encoder = resnet50
        self.encoder.fc = torch.nn.Identity()
        
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(2048, self.queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.pca = pca.PCA(threshold=threshold)
        
    def forward(self, x1, x2, **kwargs):
        y1 = self.encoder(x1)
        z1 = self.fc(y1)
        y2 = self.encoder(x2)
        z2 = self.fc(y2)
        
        # factorization
        with torch.no_grad():
            self.pca.fit(self.queue.t()) # fact_queue.t() - (queue_size, n_features)
        acts = torch.stack([y1,y2], dim=0)
        acts_centered = (acts - acts.mean(dim=0)).reshape(-1,acts.size()[-1]) # (2*batch_size,n_features)
        acts_centered_proj = self.pca.transform(acts_centered)
        self._dequeue_and_enqueue(acts.mean(dim=0))
        
        factorization = acts_centered_proj.var(dim=0).sum()/acts_centered.var(dim=0).sum()
        
        return {'out_1': z1, 'out_2': z2, 'factorization': factorization}
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = distributed.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.queue_size % batch_size == 0:  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr
        else:
            print("Incompatible batch, skipping queue update")

def get_model(*, queue_size=2048, threshold=0.9):
    model = Model(queue_size, threshold)
    
    return model

def criterion_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--sw', '--supervised-weight', type=float)
    parser.add_argument('--sc', '--supervised-criterion', type=str)
    parser.add_argument('--fw', '--factorization-weight', type=float)
    parser.add_argument('--fc', '--factorization-criterion', type=str)
    
    return parser
        
class Criterion(torch.nn.Module):
    def __init__(self, weights, criteria):
        super().__init__()
        self.weights = weights
        self.criteria = torch.nn.ModuleDict(criteria)
    
    def forward(self, input, target):
        info = {
            'supervised': self.criteria['supervised'](input['out_1'], target['label']),
            'factorization': input['factorization'],
        }
        
        loss = {
            'supervised': info['supervised'],
            'factorization': self.criteria['factorization'](
                info['factorization'],
                torch.zeros(
                    info['factorization'].shape,
                    device=info['factorization'].device
                )
            )
        }
        
        loss = sum([v1 * v2 for _, v1, v2 in utils.itertools.dict_zip(loss, self.weights, mode='strict')])
        
        return loss, info
    
def get_criterion(*, sw=1.0, fw=0.2, sc='CrossEntropyLoss', fc='L1Loss'):
    weights = {'supervised': sw, 'factorization': fw}
    criteria = {'supervised': sc, 'factorization': fc}
    criteria = {k: getattr(torch.nn, v)() for k, v in criteria.items()}
    
    criterion = Criterion(weights, criteria)
    
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
    parser.add_argument('-s', '--symmetric', action='store_true')
    parser.add_argument('-c', '--crop-size', type=int)
    parser.add_argument('--pc', '--p-color', dest='p_color', type=float)
    parser.add_argument('--pg', '--p-gray', dest='p_gray', type=float)
    parser.add_argument('--pb', '--p-blur', dest='p_blur', type=float)
    parser.add_argument('--ps', '--p-solar', dest='p_solar', type=float)
    
    return parser

def get_dataset(*, symmetric=False, crop_size=224, p_color=0.0, p_gray=0.0, p_blur=0.0, p_solar=0.0):
    transform_2 = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=p_color
        ),
        transforms.RandomGrayscale(p=p_gray),
        transforms.GaussianBlur(p=p_blur),
        transforms.Solarization(p=p_solar),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    if symmetric:
        transform_1 = transform_2
    else:
        transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    train_dataset = datasets.RandomTransformDataset(
        torchvision.datasets.ImageNet(config['paths']['imagenet'], split='train'),
        transforms.Multiplex([transform_1, transform_2]),
    )

    return train_dataset
            