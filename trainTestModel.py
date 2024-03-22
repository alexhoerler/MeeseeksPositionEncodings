import os
import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *
from transformer import Transformer
import nltk.translate.bleu_score as bleu

def createDataloaders(train_percentage=0.9, eng_source=True, batch_size=32, shuffle=True):
    if eng_source:
        if os.path.exists("engChinDataset.pth"):
            lang_dataset = torch.load("engChinDataset.pth")
        else:
            lang_dataset = ChineseEnglishDataset("seq_data.pkl")
            torch.save(lang_dataset, "engChinDataset.pth")
    else:
        if os.path.exists("chinEngDataset.pth"):
            lang_dataset = torch.load("chinEngDataset.pth")
        else:
            lang_dataset = ChineseEnglishDataset("seq_data.pkl", switchTransform=True)
            torch.save(lang_dataset, "chinEngDataset.pth")
    print(f"Dataset has {lang_dataset.__len__()} pairs.")

    train_size = int(train_percentage * len(lang_dataset))
    eval_size = len(lang_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(lang_dataset, [train_size, eval_size])
    
    train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(eval_subset, batch_size=batch_size)
    return train_data_loader, eval_data_loader

def trainModel(model, optimizer, criterion, data_loader, epoch_num, device):
    model.train()
    total_loss = 0
    
    print(f"--- Epoch {epoch_num} ---")
    for batch_num, batch in enumerate(data_loader):
        input_seq, target_seq = batch[0].to(device), batch[1].to(device)
        X, y = input_seq, target_seq

        for output_idx in range(1, y.shape[1]):

            # skip if all values to be predicted are padded values
            if torch.all(y[:, output_idx] == 0):
                break

            # filter all sequences already filtered (only padding left)
            unfinished_filter = y[:, output_idx] != 0
            y = y[unfinished_filter]
            X = X[unfinished_filter]

            # get input to decoder and true label
            y_input = y[:, :output_idx]
            y_expected = y[:, output_idx]
            
            # calculate masks
            src_pad_mask = model.create_pad_mask(X, pad_token=0).to(device)
            y_input_length = y_input.shape[1]
            tgt_mask = model.get_tgt_mask(y_input_length).to(device)

            pred = model(X, y_input, tgt_mask, src_pad_mask)
            
            pred = pred[:, -1, :] # only take the last predicted value
            loss = criterion(pred, y_expected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.detach().item()
            total_loss += loss_item

        if batch_num % 1000 == 0:
            print(f"Batch {batch_num}, Loss: {loss_item}")
    
    average_loss = total_loss / len(data_loader)
    print(f"--- Epoch {epoch_num} - Average Loss: {average_loss} ---")
    return model

def evaluateModel(model, data_loader, device):
    model.eval()
    total_bleu = 0
    
    outputs = []
    with torch.no_grad():
        for batch in data_loader:
            input_seq, target_seq = batch[0].to(device), batch[1].to(device)
            X, y = input_seq, target_seq

            y_input = y[:, :1]
            for output_idx in range(1, y.shape[1]):
                unfinished_filter = y_input[:, -1] != 102 # 102 is [EOS] token number for BERT tokenizers
                finished_filter = ~unfinished_filter
                y_finished = y_input[finished_filter]
                for row_idx in range(y_finished.shape[0]):
                    outputs.append(([y[row_idx].tolist()], y_finished[row_idx].tolist()))
                
                y = y[unfinished_filter]
                y_input = y_input[unfinished_filter]
                X = X[unfinished_filter]

                src_pad_mask = model.create_pad_mask(X, pad_token=0).to(device)
                y_input_length = y_input.shape[1]
                tgt_mask = model.get_tgt_mask(y_input_length).to(device)

                pred = model(X, y_input, tgt_mask, src_pad_mask)
                pred = pred[:, -1, :] # only take the last predicted value
                tokens = torch.argmax(pred, dim=-1)
                y_input = torch.cat((y_input, tokens), dim=-1)
            
            for row_idx in range(y_input.shape[0]):
                outputs.append(([y[row_idx].tolist()], y_input[row_idx].tolist()))
    
    for true_label_list, generated in outputs:
        try:
            total_bleu += bleu.sentence_bleu(true_label_list, generated, weights=(0.33, 0.33, 0.33, 0))
        except:
            total_bleu += 0
    
    return total_bleu / len(outputs)

if __name__ == "__main__":
    train_data_loader, eval_data_loader = createDataloaders()
    eng_vocab_size = engBertTokenizer.vocab_size
    chin_vocab_size = chinBertTokenizer.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    
    model = Transformer(input_vocab_size=eng_vocab_size, output_vocab_size=chin_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 20

    for epoch_num in range(epochs):
        model = trainModel(model, optimizer, criterion, train_data_loader, epoch_num, device)
        average_bleu = evaluateModel(model, eval_data_loader)
        print(f"Epoch {epoch_num} - Average Bleu: {average_bleu}")
        torch.save(model.state_dict(), "transformer.pth")