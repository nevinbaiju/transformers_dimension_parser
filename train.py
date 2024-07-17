from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch

from dataset import DescriptionDataset
from model import Model

from tqdm import tqdm
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(model, data_loader):
    with torch.no_grad():
        loss = 0
        for desc, dims in tqdm(data_loader):
            dims = dims.to(device).reshape(64, -1, 10).argmax(dim=2)
            dims_pred = model(desc.to(device)).reshape(64, -1, 10).argmax(dim=2)

            loss += (torch.abs(dims - dims_pred)).cpu().numpy().sum()

    return loss/(len(data_loader)*64)

def train():
    model = Model(10)
    model = model.to(device)
    
    train = DescriptionDataset()
    test = DescriptionDataset(train=False)
    train_data_loader = DataLoader(train, batch_size=64)
    test_data_loader = DataLoader(test, batch_size=64)
    
    num_epochs = 40
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        losses = []

        for desc, dims in tqdm(train_data_loader):
            optimizer.zero_grad()
            
            desc = desc.to(device)
            dims = dims.to(device)

            dims_pred = model(desc)
            loss = cross_entropy(dims_pred, dims)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch + 1,
                np.mean(losses),
            ))

        if (epoch+1) % 5 == 0:
            print("Validating...")
            train_mae = eval(model, train_data_loader)
            test_mae = eval(model, test_data_loader)

            print(f'MAE after {epoch+1} epochs: \nTrain: {train_mae}\nTest: {test_mae}')
            torch.save(model, f'models/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.mkdir('models')
    train()