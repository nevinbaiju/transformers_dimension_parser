import pandas as pd
from torch.utils.data.dataset import Dataset
import torch
from config import vec_len

class DescriptionDataset(Dataset):
    def __init__(self, train=True):
        self.data = pd.read_csv('data.csv')
        split = int(0.8*len(self.data))
        if train:
            self.data = self.data.iloc[:split, :]
        else:
            self.data = self.data.iloc[split:, :]
    
    def __len__(self):
        return len(self.data)

    def encode(self, description):
        encoding = [0]*vec_len
        for i,c in enumerate(description):
            if c.isdigit():
                encoding[i] = int(c)+1
            elif c == 'X':
                encoding[i] = 11
            else:
                encoding[i] = 12
        return encoding
    
    def __getitem__(self, index):
        data = self.data.iloc[index]
        description = self.encode(data['description'])
        
        height = data['height']
        width = data['width']
        y = [[0]*10 for i in range(4)]
        y[1][height%10] = 1
        height //= 10
        y[3][width%10] = 1
        width //= 10
        y[0][height%10] = 1
        y[2][height%10] = 1
        
        return torch.Tensor(description).to(torch.int), torch.Tensor(y).flatten()
