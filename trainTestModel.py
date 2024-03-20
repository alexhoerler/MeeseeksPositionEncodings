import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *

def createDataloader(eng_source=True, batch_size=32, shuffle=True):
    if eng_source:
        lang_dataset = ChineseEnglishDataset("seq_data.pkl")
    else:
        lang_dataset = ChineseEnglishDataset("seq_data.pkl", switchTransform=True)
    
    data_loader = DataLoader(lang_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

