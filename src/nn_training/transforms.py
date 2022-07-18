#######################################################
### IMPORTANT: ONLY WORKS FOR TORCHVISION >= 0.11.0 ###
#######################################################

from abc import ABC, abstractmethod
import random

import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from PIL import ImageOps, ImageFilter

InterpolationMode = T.InterpolationMode

class LabeledTransform(ABC):
    @abstractmethod
    def __call__(self, img, params):
        # Transforms an img with params (same as torchvision.transform classes)
        pass
    
    @abstractmethod
    def get_params(self, img):
        pass
    
    @abstractmethod
    def get_target(self, img, params):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass

### Subclasses of torch.nn.module (same as tochvision.transform classes) ###

Normalize = T.Normalize     
ToTensor = T.ToTensor

class GaussianBlur(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

### Subclasses of LabeledTransform ###

class RandomApply(LabeledTransform):
    def __init__(self, transforms, **kwargs):
        self.transform = T.RandomApply(transforms, **kwargs)
    
    def __call__(self, img, params):
        if self.transform.p < params['p']:
            return img
        for t in self.transform.transforms:
            img = t(img, params[repr(t)])
        return img
    
    def get_params(self, img):
        params = {'p': torch.rand(1)}
        other_params = {repr(t): t.get_params(img) for t in self.transform.transforms}
        return {**params, **other_params}
    
    def get_target(self, img, params):
        transforms_name = '_'.join([transform.__class__.__name__ for transform in self.transform.transforms])
        target = {
            f'applied_{transforms_name}': torch.Tensor([params['p'] <= self.transform.p]), # 1 if True, 0 if False
            **{k: v for t in self.transform.transforms for k, v in t.get_target(img, params[repr(t)]).items()},
        }
        return target
    
    def __repr__(self):
        return repr(self.transform)
    
class RandomGrayscale(LabeledTransform):
    def __init__(self, **kwargs):
        self.transform = T.RandomGrayscale(**kwargs)
        
    def __call__(self, img, params):
        num_output_channels = F.get_image_num_channels(img)
        if params[0] < self.transform.p:
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        return img
    
    def get_params(self, img):
        return (torch.rand(1),)
    
    def get_target(self, img, params):
        target = {
            'applied_RandomGrayscale': torch.Tensor([params[0] < self.transform.p]), # 1 if True, 0 if False
        }
        return target
    
    def __repr__(self):
        return repr(self.transform)
    
class RandomHorizontalFlip(LabeledTransform):
    def __init__(self, **kwargs):
        self.transform = T.RandomHorizontalFlip(**kwargs)
        
    def __call__(self, img, params):
        if params[0] < self.transform.p:
            return F.hflip(img)
        return img
    
    def get_params(self, img):
        return (torch.rand(1),)
    
    def get_target(self, img, params):
        target = {
            'applied_RandomHorizontalFlip': torch.Tensor([params[0] < self.transform.p]), # 1 if True, 0 if False
        }
        return target
    
    def __repr__(self):
        return repr(self.transform)

class RandomResizedCrop(LabeledTransform):
    def __init__(self, *args, **kwargs):
        self.transform = T.RandomResizedCrop(*args, **kwargs)
    
    def __call__(self, img, params):
        return F.resized_crop(img, *params, self.transform.size, self.transform.interpolation)
    
    def get_params(self, img):
        return self.transform.get_params(img, self.transform.scale, self.transform.ratio)
    
    def get_target(self, img, params):
        top, left, height, width = params
        im_width, im_height = img.size
        v_scale, h_scale = height/im_height, width/im_width
        scale = max(v_scale, h_scale)
        target = {
            'cam_pos_y': torch.Tensor([top + height/2]),
            'cam_pos_x': torch.Tensor([left + width/2]),
            'cam_scale': torch.Tensor([scale]),
        }
        return target
    
    def __repr__(self):
        return repr(self.transform)
    
class ColorJitter(LabeledTransform):
    def __init__(self, *args, **kwargs):
        self.transform = T.ColorJitter(*args, **kwargs)
        
    def __call__(self, img, params):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img
    
    def get_params(self, img):
        return self.transform.get_params(self.transform.brightness, self.transform.contrast, self.transform.saturation, self.transform.hue)
        
    def get_target(self, img, params):
        target = {
            'brightness': torch.Tensor([params[1]]),
            'contrast': torch.Tensor([params[2]]),
            'saturation': torch.Tensor([params[3]]),
            'hue': torch.Tensor([params[4]]),
        }
        return target
    
    def __repr__(self):
        return repr(self.transform)
    
class Compose(LabeledTransform):
    """
    Composes multiple torch.nn.module/LabeledTransforms
    """
    def __init__(self, transforms):
        self.transforms = transforms
        self.labeled_transforms = [transform for transform in self.transforms if isinstance(transform, LabeledTransform)]
        
    def __call__(self, img, params):
        for transform in self.transforms:
            if transform in self.labeled_transforms:
                img = transform(img, params[repr(transform)])
            else:
                img = transform(img)
        return img
    
    def get_params(self, img):
        return {repr(transform): transform.get_params(img) for transform in self.labeled_transforms}
    
    def get_target(self, img, params):
        target = {k: v for transform in self.labeled_transforms for k, v in transform.get_target(img, params[repr(transform)]).items()}
        return target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class Multiplex(LabeledTransform):
    """
    Multiplexes multiple LabeledTransforms
    """
    def __init__(self, transforms):
        assert all([isinstance(transform, LabeledTransform) for transform in transforms])
        self.transforms = transforms
        
    def __call__(self, img, params):
        return [transform(img, param) for transform, param in zip(self.transforms, params)]
    
    def get_params(self, img):
        return [transform.get_params(img) for transform in self.transforms]
    
    def get_target(self, img, params):
        return {f'{k}_{i}': v for i, (transform, param) in enumerate(zip(self.transforms, params)) for k, v in transform.get_target(img, param).items()}
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
        

class Siamese(LabeledTransform):
    def __init__(self, transform):
        self.transform = transform
        self.is_labeled_transform = isinstance(self.transform, LabeledTransform)
        
    def __call__(self, img, params):
        if self.is_labeled_transform:
            return [self.transform(img, params[0]), self.transform(img, params[1])]
        return [self.transform(img), self.transform(img)]
    
    def get_params(self, img):
        if self.is_labeled_transform:
            return [self.transform.get_params(img), self.transform.get_params(img)]
        return []
    
    def get_target(self, img, params):
        if self.is_labeled_transform:
            target_0, target_1 = self.transform.get_target(img, params[0]), self.transform.get_target(img, params[1])
            return {**{f'{k}_0': v for k, v in target_0.items()}, **{f'{k}_1': v for k, v in target_1.items()}}
        return {}
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += '    {0}'.format(self.transform)
        format_string += '\n)'
        return format_string