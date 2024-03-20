import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *

def createDataloaders(train_percentage=0.9, eng_source=True, batch_size=32, shuffle=True):
    if eng_source:
        lang_dataset = ChineseEnglishDataset("seq_data.pkl")
    else:
        lang_dataset = ChineseEnglishDataset("seq_data.pkl", switchTransform=True)
    
    train_size = int(train_percentage * len(lang_dataset))
    eval_size = len(lang_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(lang_dataset, [train_size, eval_size])
    
    train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(eval_subset, batch_size=batch_size)
    return train_data_loader, eval_data_loader

