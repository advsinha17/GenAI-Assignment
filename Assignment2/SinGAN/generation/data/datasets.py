import torch
import random
from math import floor
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', batch_size=1, crop_size=0):
        self.root = root
        self.batch_size = batch_size
        self.crop_size = crop_size
        self._init()

    def _init(self):
        # to tensor
        self.to_tensor = transforms.ToTensor()
        # open image
        image = Image.open(self.root).convert('RGB')
        self.image = self.to_tensor(image).unsqueeze(dim=0)
        self.image = (self.image - 0.5) * 2
        print(f'Loading image from {self.root}...',flush=True)
        print(f"shape of the image is: ", self.image.shape,flush=True)
        
        # set from outside
        self.reals = None
        self.noises = None
        self.amps = None

    def __getitem__(self, index):
        amps = self.amps
        reals = self.reals 
        noises = self.noises 

        return {'reals': reals, 'noises': noises, 'amps': amps}
       
    def __len__(self):
        return self.batch_size