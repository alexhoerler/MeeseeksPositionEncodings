import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

engBertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
chinBertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class seqTokenizer():
    def __init__(self):
        self.token_to_idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[EOS]": 3}
        self.idx_to_token = None
        self.frequency_dict = dict()
        self.length = 4
    
    def __len__(self):
        return self.length
    
    def addToken(self, token):
        frequency = self.frequency_dict.get(token, 0)
        self.frequency_dict[token] = frequency + 1
    
    def addSeq(self, seq, splitSpace=True):
        seq = seq.replace(",", "").replace("，", "").replace(".", "").replace("。", "").replace("?", "").replace("？", "").replace("!", "").replace("！", "").strip(" -")
        if splitSpace:
            seq_tokens = seq.split(" ")
        else:
            seq_tokens = list(seq)
        for token in seq_tokens:
            sub_tokens = token.split("'")
            for sub_idx, sub_token in enumerate(sub_tokens):
                if sub_idx == 0:
                    self.addToken(sub_token)
                else:
                    self.addToken(f"'{sub_token}")
    
    def create_dicts(self):
        for token, frequency in self.frequency_dict.items():
            if frequency > 3:
                if self.token_to_idx.get(token) is None:
                    self.token_to_idx[token] = self.length
                    self.length += 1
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
    
    def seq_to_idx(self, seq, splitSpace=True):
        tokens = [self.token_to_idx["[CLS]"]]
        seq = seq.replace(",", "").replace("，", "").replace(".", "").replace("。", "").replace("?", "").replace("？", "").replace("!", "").replace("！", "").strip(" -")
        if splitSpace:
            seq_tokens = seq.split(" ")
        else:
            seq_tokens = list(seq)
        for token in seq_tokens:
            sub_tokens = token.split("'")
            for sub_idx, sub_token in enumerate(sub_tokens):
                if sub_idx == 0:
                    tokens.append(self.token_to_idx.get(sub_token, 1))
                else:
                    tokens.append(self.token_to_idx.get(f"'{sub_token}", 1))
        tokens.append(self.token_to_idx["[EOS]"])
        return tokens
    
    def idx_to_seq(self, idx, includeInvis=False, splitSpace=True):
        if not isinstance(idx, list):
            idx = idx.tolist()
        seq = ""
        for i in idx:
            seq += self.idx_to_token.get(i, "[UNK]")
            if splitSpace:
                seq += " "

        if includeInvis:
            return seq
        else:
            return seq.replace("[PAD]", "").replace("[CLS]", "").replace("[EOS]", "").replace(" '", "'").strip()

# Define Dataset
class ChineseEnglishDataset(Dataset):
    def __init__(self, englishChinesePickle, eng_max_length=10, chin_max_length=10):
        '''
        Initialize Dataset
        Params:
            englishChinesePickle: {row_number: (source sequence, target sequence)}
            switchTransform: False for English->Chinese, True for Chinese->English
            englishTokenizer: English language Tokenizer to use
            chineseTokenizer: Chinese language Tokenizer to use
        '''
        eng_chin_dict = dict()
        with open(englishChinesePickle, "rb") as pickle_file:
            eng_chin_dict = pickle.load(pickle_file)
        
        self.seqTokenizer = seqTokenizer()
        self.seq_max_length = eng_max_length
        self.targetTokenizer = seqTokenizer()
        self.target_max_length = chin_max_length

        self.sequences = []
        self.targets = []

        for key_number in eng_chin_dict:
            mappings = eng_chin_dict[key_number]
            if len(mappings[0]) < 35 and len(mappings[1]) < 15:
                self.sequences.append(mappings[0])
                self.seqTokenizer.addSeq(mappings[0])
                self.targets.append(mappings[1])
                self.targetTokenizer.addSeq(mappings[1], splitSpace=False)
            if len(self.targets) >= 6000 * 32:
                break
        
        assert len(self.sequences) == len(self.targets)
        self.seqTokenizer.create_dicts()
        self.targetTokenizer.create_dicts()
        
        self.seq_tensors = []
        self.target_tensors = []
        for seq_number, seq_sent in enumerate(self.sequences):
            seq_tokens = self.seqTokenizer.seq_to_idx(seq_sent)
            tgt_sent = self.targets[seq_number]
            target_tokens = self.targetTokenizer.seq_to_idx(tgt_sent, splitSpace=False)
            if seq_tokens.count(1) > 1 or target_tokens.count(1) > 1:
                continue

            if len(seq_tokens) > self.seq_max_length:
                seq_tokens = seq_tokens[: self.seq_max_length - 1] + seq_tokens[-1: ]
            else:
                seq_tokens = seq_tokens + [0] * (self.seq_max_length - len(seq_tokens))
            
            if len(target_tokens) > self.target_max_length:
                target_tokens = target_tokens[: self.target_max_length - 1] + target_tokens[-1: ]
            else:
                target_tokens = target_tokens + [0] * (self.target_max_length - len(target_tokens))
            
            self.seq_tensors.append(torch.tensor(seq_tokens))
            self.target_tensors.append(torch.tensor(target_tokens))

        assert len(self.target_tensors) == len(self.seq_tensors)
    
    def __len__(self):
        return len(self.target_tensors)
    
    def __getitem__(self, index):
        return self.seq_tensors[index], self.target_tensors[index]
