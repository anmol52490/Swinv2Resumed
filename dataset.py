import numpy as np
import torch
from torch.utils.data import Dataset

class FoodSegDataset(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        self.hf_dataset = hf_dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        image = np.array(item['image'].convert("RGB"))
        mask = np.array(item['label']) 
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            mask = mask.long() if hasattr(mask, 'long') else torch.tensor(mask, dtype=torch.long)
            
        return image, mask