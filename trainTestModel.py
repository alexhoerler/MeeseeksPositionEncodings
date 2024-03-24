'''
standard - epochs: 1
'''
import os
import argparse
import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *
from transformerModel import TransformerModel
import nltk.translate.bleu_score as bleu

def createDataloaders(train_percentage=0.95, eng_source=True, batch_size=32, shuffle=False, on_pace=False):
    if on_pace:
        location_prefix = "/storage/home/hcoda1/9/ahoerler3/scratch/p-ahoerler3-1/MeeseeksPositionEncodings/"
    else:
        location_prefix = ""

    if eng_source:
        dataset_file = "engChinDataset.pth"
        dataset_path = f"{location_prefix}{dataset_file}"
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_file} file")
            lang_dataset = torch.load(dataset_path)
        else:
            print(f"Creating dataset from pickle file")
            lang_dataset = ChineseEnglishDataset("seq_data.pkl")
            torch.save(lang_dataset, dataset_path)
    else:
        dataset_file = "chinEngDataset.pth"
        dataset_path = f"{location_prefix}{dataset_file}"
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_file} file")
            lang_dataset = torch.load(dataset_file)
        else:
            print(f"Creating dataset from pickle file")
            lang_dataset = ChineseEnglishDataset("seq_data.pkl", switchTransform=True)
            torch.save(lang_dataset, dataset_path)
    print(f"Dataset loaded; has {lang_dataset.__len__()} pairs.")

    train_size = int(train_percentage * len(lang_dataset))
    eval_size = len(lang_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(lang_dataset, [train_size, eval_size])
    
    train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(eval_subset, batch_size=batch_size)
    return train_data_loader, eval_data_loader

def trainModel(model, optimizer, criterion, data_loader, epoch_num, device, start_batch, model_path):
    model.train()
    total_loss = 0
    
    print(f"--- Epoch {epoch_num} ---")
    for batch_num, batch in enumerate(data_loader):
        if epoch_num == 0 and batch_num < start_batch:
            continue

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

        if batch_num % 10000 == 0:
            print(f"Batch {batch_num}, Loss: {loss_item}")
            torch.save(model.state_dict(), model_path)
    
    average_loss = total_loss / len(data_loader)
    print(f"--- Epoch {epoch_num} - Average Loss: {average_loss} ---")
    return model

def evaluateModel(model, data_loader, device):
    model.eval()
    total_bleu = 0
    
    outputs = []
    with torch.no_grad():
        for batch_number, batch in enumerate(data_loader):
            input_seq, target_seq = batch[0].to(device), batch[1].to(device)
            X, y = input_seq, target_seq

            y_input = y[:, :1]
            for output_idx in range(1, y.shape[1]):
                # skip if all values to be predicted are padded values
                if torch.all(y_input[:, -1] == 102):
                    print("breaking")
                    break

                # filter all sequences already filtered (generated EOS)
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
                tokens = torch.argmax(pred, dim=-1, keepdim=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--on_pace", help="Whether it's running on a Pace cluster", action="store_true")
    parser.add_argument("-e", "--eng_src", help="Whether it's english source", action="store_true")
    parser.add_argument("-m", "--model_name", type=str, nargs='?', help="The model name")
    parser.add_argument("-s", "--start_batch", type=int, help="The batch to start with")
    args = parser.parse_args()
    if args.on_pace:
        location_prefix = "/storage/home/hcoda1/9/ahoerler3/scratch/p-ahoerler3-1/MeeseeksPositionEncodings/"
        print("Intended for Pace")
    else:
        location_prefix = ""

    eng_vocab_size = engBertTokenizer.vocab_size
    chin_vocab_size = chinBertTokenizer.vocab_size
    if args.eng_src:
        print("English as source, Chinese as target")
        in_vocab_size = eng_vocab_size
        out_vocab_size = chin_vocab_size
    else:
        print("Chinese as source, English as target")
        in_vocab_size = chin_vocab_size
        out_vocab_size = eng_vocab_size
    
    if args.start_batch:
        start_batch = args.start_batch
    else:
        start_batch = 0

    model_path = f"{location_prefix}{args.model_name}.pth"
    train_data_loader, eval_data_loader = createDataloaders(eng_source=args.eng_src, on_pace=args.on_pace)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    
    model = TransformerModel(input_vocab_size=in_vocab_size, output_vocab_size=out_vocab_size).to(device)
    if os.path.exists(model_path):
        print("Loading model from file")
        model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 20

    print("\nStarting Training/Testing")
    for epoch_num in range(epochs):
        model = trainModel(model, optimizer, criterion, train_data_loader, epoch_num, device, start_batch, model_path)
        torch.save(model.state_dict(), model_path)
        average_bleu = evaluateModel(model, eval_data_loader, device)
        print(f"Epoch {epoch_num} - Average Bleu: {average_bleu}")