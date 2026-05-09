import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class OCRDataset(Dataset):
    def __init__(self,folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char2idx = {c:i+1 for i,c in enumerate(self.chars)}

    def __len__(self):
        return len(self.paths)

    def encode(self, text):
        return [self.char2idx[c] for c in text]

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L').resize((128,32))
        img = torch.tensor(list(img.getdata()),dtype=torch.float32).reshape(1,32,128)/255.0

        label = os.path.basename(path).split('_')[0]
        label_encoded = torch.tensor(self.encode(label),dtype=torch.long)

        return img, label_encoded