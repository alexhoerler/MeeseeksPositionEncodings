import pickle
import torch
from torch.utils.data import Dataset

# Define Dataset
class ChineseEnglishDataset(Dataset):
    def __init__(self, englishChinesePickle, switchTransform=False):
        eng_chin_dict = dict()
        with open(englishChinesePickle, "rb") as pickle_file:
            eng_chin_dict = pickle.load(pickle_file)
        
        self.switchSeqLabel = switchTransform
        self.sequences = []
        self.labels = []
        for key_number in eng_chin_dict:
            mappings = eng_chin_dict[key_number]
            self.sequences.append(mappings[0])
            self.labels.append(mappings[1])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if not self.switchSeqLabel:
            return self.sequences[index], self.labels[index]
        else:
            return self.labels[index], self.sequences[index]
