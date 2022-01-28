'''
Training code

Written by:
    Simo Ryu
'''

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from eq_dataset import EquationsDataset
from models import SOP


def train():
    device = torch.device("cuda:0")
    epochs = 1
    batch_size = 64
    lr = 1e-4
    max_len = 128
    
    #chars = list("0987654321-+*()^xyz") (case with three variables)
    chars = list("0987654321-+*()^xy")
    n_vocab = len(chars) + 2
    model = SOP(
        d_model = 512,
        n_head = 8,
        num_layers = 6,
        n_vocab = n_vocab, 
        max_len = max_len, 
        chars = chars,
        device = device
    )
    opt = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-10)
    dataset = EquationsDataset(max_len = max_len, chars = chars)
    dl = DataLoader(dataset, shuffle= True, batch_size= batch_size,  drop_last= True, num_workers = 3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl)
        tot_loss = 0
        cnt = 0
        for (x, yin, yout) in pbar:
            
            x = x.to(device)
            yin = torch.cat([torch.ones(batch_size, 1) * (n_vocab - 1), yin], dim = 1).long()
            yin = yin.to(device)
            yout = yout.to(device)
            y_pred = model(x, yin)

            loss = criterion(y_pred.view(-1, n_vocab - 1), yout.view(-1))
            model.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            cnt += 1
            pbar.set_description(f"current loss : {tot_loss/cnt:.5f}")

        eq = "2*y^4-2*y^3-y^2+1"
        ans = "(1-y^2)^2+(-y^2+y)^2"

        ral = model.toSOP(eq, gen_len = max_len - 1)
        print(f'Epoch {epoch} : Loss : {tot_loss/cnt :.5f}, Example : {ral[0]}')

    torch.save(model, "model.dat")


if __name__ == "__main__":
    train()