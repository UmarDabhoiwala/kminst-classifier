import os
import numpy as np
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    base_filename = 'kmnist-{}-{}.npz'
    data_filepart = 'imgs'
    labels_filepart = 'labels'
    
    def __init__(self, folder, train_or_test='train', transforms=None):
        self.root = os.path.expanduser(folder)
   

        
        self.data = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.data_filepart)))['arr_0']
        self.targets = np.load(os.path.join(self.root, self.base_filename.format(train_or_test, self.labels_filepart)))['arr_0']

        self.transforms = transforms
    
    def __getitem__(self, index):
      
        cur_data = np.expand_dims(self.data[index], axis=-1)

        if self.transforms:
            cur_data = self.transforms(cur_data)
        
        target = int(self.targets[index])
        img, target = cur_data, target
        
        return img, target

    def __len__(self):
        return len(self.data)