import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *
from transformer import Transformer

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

def trainModel(model, optimizer, criterion, data_loader, iteration_num, device):
    model.train()
    total_loss = 0
    
    print(f"--- Iteration {iteration_num} ---")
    for batch_num, batch in enumerate(data_loader):
        input_seq, target_seq = batch[0].to(device), batch[1].to(device)
        X, y = input_seq, target_seq

        for output_idx in range(1, y.shape[1]):

            # skip if all values to be predicted are padded values
            if torch.all(y[:, output_idx] == 0):
                break

            # filter all sequences already filtered (only padding left)
            finished_filter = y[:, output_idx] != 0
            y = y[finished_filter]
            X = X[finished_filter]

            # get input to decoder and true label
            y_input = y[:, :output_idx]
            y_expected = y[:, output_idx]
            
            # calculate masks
            src_pad_mask = model.create_pad_mask(X, pad_token=0).to(device)
            y_input_length = y_input.shape[1]
            tgt_mask = model.get_tgt_mask(y_input_length).to(device)

            pred = model(X, y_input, tgt_mask, src_pad_mask)
            
            pred = pred.permute(1, 2, 0)
            loss = criterion(pred, y_expected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.detach().item()
            total_loss += loss_item

        if batch_num % 100000 == 0:
            print(f"Batch {batch_num}, Loss: {loss_item}")
    
    average_loss = total_loss / len(data_loader)
    print(f"--- Iteration {iteration_num} - Average Loss: {average_loss} ---")
    return model

def evaluateModel(model, data_loader):
    model.eval()
    total_bleu = 0
    total_seqs = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_seq, target_seq = batch[0].to(device), batch[1].to(device)
            X, y = input_seq, target_seq
    
    return total_bleu / total_seqs

if __name__ == "__main__":
    train_data_loader, eval_data_loader = createDataloaders()
    eng_vocab_size = engBertTokenizer.vocab_size
    chin_vocab_size = chinBertTokenizer.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(input_vocab_size=eng_vocab_size, output_vocab_size=chin_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10

    for epoch_num in range(epochs):
        model = trainModel(model, optimizer, criterion, train_data_loader, epoch_num, device)
        torch.save(model.state_dict(), "transformer.pth")