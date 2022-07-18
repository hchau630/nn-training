import torch
from torchvision.datasets import ImageFolder

class RandomTransformDataset(torch.utils.data.Dataset):
    """
    A dataset together with custom random transforms
    whose data are the transformed images and the labels are the transform values 
    
    Inputs:
    dataset - a torch.utils.data.Dataset instance
    transform - a LabeledTransform instance
    """
    def __init__(self, dataset, transform, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        params = self.transform.get_params(image)
        target = self.transform.get_target(image, params)
        image = self.transform(image, params) # transform images
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, dict(label=label, **target)

    def __len__(self):
        return len(self.dataset)