import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(Dataset):
    def __init__(self,phase,imgarr,labelarr,transforms=None):
        self.phase = phase
        self.imgarr = imgarr
        self.labelarr = labelarr
        self.transforms=transforms
        
        
    def __len__(self):
        return len(self.imgarr)
    
    def __getitem__(self,idx):
        
        x = self.imgarr[idx] 
        y = self.labelarr[idx]
        #x = x.astype(np.float32)/255.0

        x1 = self.augment(x)
        x2 = self.augment(x)
        
        #x1 = self.preprocess(x1)
        #x2 = self.preprocess(x2)
        
        return x1, x2,y

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.imgarr = self.imgarr[random.sample(population = list(range(self.__len__())),k = self.__len__())]

        
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame, transformations = None):
        
        if self.transforms is not None:
            frame = self.transforms(frame)
        
        return frame
        

class BaseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels, transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample=self.imgs[idx]
        y=self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample,y