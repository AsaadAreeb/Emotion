import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image



class FERDataset(Dataset):
    def __init__(self, csv_data, transform, train=True):
        self.data = pd.read_csv(csv_data)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            pixels = self.data.iloc[idx, 1].split()
            pixels = np.array(pixels, dtype=np.uint8).reshape(48, 48)
    
            image = Image.fromarray(pixels)
    
            label = int(self.data.iloc[idx, 0])
    
            if self.transform:
                image = self.transform(image)
    
            return image, label

        pixels = self.data.iloc[idx, 0].split()
        pixels = np.array(pixels, dtype=np.uint8).reshape(48, 48)

        image = Image.fromarray(pixels)

        if self.transform:
            image = self.transform(image)

        return image
    


def generate_TrainValTest_dataloaders(transform, trpath='', tstpath='', batch_size=32):
    Train_fer_dataset = FERDataset(csv_data=trpath, transform=transform)
    
    train_size = int(0.8 * len(Train_fer_dataset))
    val_size = len(Train_fer_dataset) - train_size

    Train_fer_dataset, Val_fer_dataset = random_split(Train_fer_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    Test_fer_dataset = FERDataset(csv_data=tstpath, transform=transform, train=False)

    TrainDataLoader = DataLoader(Train_fer_dataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(Val_fer_dataset, batch_size=batch_size)
    TestDataLoader = DataLoader(Test_fer_dataset, batch_size=batch_size)
    return TrainDataLoader, ValDataLoader, TestDataLoader

