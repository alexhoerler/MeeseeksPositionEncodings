import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

engBertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
chinBertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Define Dataset
class ChineseEnglishDataset(Dataset):
    def __init__(self, englishChinesePickle, switchTransform=False, englishTokenizer=engBertTokenizer, chineseTokenizer=chinBertTokenizer, max_length=25):
        eng_chin_dict = dict()
        with open(englishChinesePickle, "rb") as pickle_file:
            eng_chin_dict = pickle.load(pickle_file)
        
        self.switchSeqTarget = switchTransform
        if not self.switchSeqTarget:
            self.seqTokenizer = englishTokenizer
            self.targetTokenizer = chineseTokenizer
        else:
            self.seqTokenizer = chineseTokenizer
            self.targetTokenizer = englishTokenizer
        self.max_length = max_length

        self.sequences = []
        self.targets = []

        if not self.switchSeqTarget:
            for key_number in eng_chin_dict:
                mappings = eng_chin_dict[key_number]
                self.sequences.append(mappings[0])
                self.targets.append(mappings[1])
        else:
            for key_number in eng_chin_dict:
                mappings = eng_chin_dict
                self.sequences.append(mappings[1])
                self.targets.append(mappings[0])
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        seq_tokens = self.seqTokenizer(self.sequences[index])
        if len(seq_tokens) > self.max_length:
            seq_tokens = seq_tokens[: self.max_length - 1] + seq_tokens[-1: ]
        else:
            seq_tokens = seq_tokens + [0] * (self.max_length - len(seq_tokens))
        
        target_tokens = self.targetTokenizer(self.targets[index])
        if len(target_tokens) > self.max_length:
            target_tokens = target_tokens[: self.max_length - 1] + target_tokens[-1: ]
        else:
            target_tokens = target_tokens + [0] * (self.max_length - len(target_tokens))
        
        return torch.tensor(seq_tokens), torch.tensor(target_tokens)
