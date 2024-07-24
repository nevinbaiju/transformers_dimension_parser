from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch

from dataset import DescriptionDataset
from model import Model
from config import n_blocks

from tqdm import tqdm
import numpy as np
import os
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(model, data_loader):
    with torch.no_grad():
        loss = 0
        inc_arr = []
        for desc, dims in tqdm(data_loader):
            dims = dims.to(device).reshape(64, -1, 10).argmax(dim=2)
            dims_pred = model(desc.to(device)).reshape(64, -1, 10).argmax(dim=2)
            inc_arr.append((dims != dims_pred).cpu().numpy().mean())
            loss += (torch.abs(dims - dims_pred)).cpu().numpy().sum()

    return np.mean(inc_arr), loss/(len(data_loader)*64)

def train():
    model = Model(n_blocks)
    model = model.to(device)
    
    train = DescriptionDataset()
    test = DescriptionDataset(train=False)
    train_data_loader = DataLoader(train, batch_size=64)
    test_data_loader = DataLoader(test, batch_size=64)
    
    num_epochs = 40
    optimizer = Adam(model.parameters(), lr=1e-3)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_log_file = f'train_logs/loss_log_{ts}.txt'
    metrics_log_file = f'train_logs/metrics_log_{ts}.txt'

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

        total_loss = np.mean(losses)
        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch + 1,
                total_loss,
            ))
        with open(loss_log_file, 'a') as file:
            file.write(f'{total_loss}\n')

        if (epoch+1) % 5 == 0:
            print("Validating...")
            train_err, train_mae = eval(model, train_data_loader)
            test_err, test_mae = eval(model, test_data_loader)

            print(f'MAE after {epoch+1} epochs: \nTrain:\n MAE: {train_mae}, Acc: {1-train_err}\nTest\n MAE: {test_mae}, Acc: {1-test_err}')
            torch.save(model, f'models/model_{ts}_epoch_{epoch+1}.pth')
            with open(metrics_log_file, 'a') as file:
                file.write(f'{train_mae},{train_err},{test_mae},{test_err}\n')

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('train_logs'):
        os.mkdir('train_logs')
    train()