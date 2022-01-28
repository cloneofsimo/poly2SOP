from typing import List
import torch
from torch.utils.data import Dataset, DataLoader


def _load_data(file_path, chars, max_len):
    atoi = {c: i + 1 for i, c in enumerate(chars)}

    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        ints = [atoi[c] for c in line] + [0] * max_len
        ints = ints[:max_len]
        data.append(ints)

    return torch.tensor(data, dtype=torch.long)

class EquationsDataset(Dataset):
    def __init__(self, max_len : int, chars : List[str]):
        super().__init__()
        # since data is small, load in memory
        self.x = _load_data('train_x.txt', chars, max_len)
        self.y = _load_data('train_y.txt', chars, max_len)

        self.max_len = max_len

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        _x = self.x[index]
        _y = self.y[index]
        return _x, _y[:-1], _y

