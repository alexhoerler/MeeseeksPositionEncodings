'''
standard - epochs: 2
'''
import os
import argparse
import torch
from torch.utils.data import DataLoader
from chineseEnglishDataset import *
from transformerModel import TransformerModel
import nltk.translate.bleu_score as bleu

seqTokenizer = None
targetTokenizer = None

def createDataloaders(train_percentage=0.96, batch_size=32, shuffle=False, on_pace=False):
    if on_pace:
        location_prefix = "/storage/home/hcoda1/9/ahoerler3/scratch/p-ahoerler3-1/MeeseeksPositionEncodings/"
    else:
        location_prefix = ""

    dataset_file = "engChinDataset.pth"
    dataset_path = f"{location_prefix}{dataset_file}"
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_file} file")
        lang_dataset = torch.load(dataset_path)
    else:
        print(f"Creating dataset from pickle file")
        lang_dataset = ChineseEnglishDataset("seq_data.pkl")
        torch.save(lang_dataset, dataset_path)
    print("Sequence Tokenizer length: ", len(lang_dataset.seqTokenizer))
    print("Target Tokenizer length: ", len(lang_dataset.targetTokenizer))
    global seqTokenizer
    seqTokenizer = lang_dataset.seqTokenizer
    global targetTokenizer
    targetTokenizer = lang_dataset.targetTokenizer
    print(f"Dataset loaded; has {len(lang_dataset)} pairs.")

    train_size = int(train_percentage * len(lang_dataset))
    eval_size = len(lang_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(lang_dataset, [train_size, eval_size])
    
    train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    eval_data_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=shuffle)
    return train_data_loader, eval_data_loader

def trainModel(model, criterion, optimizer, data_loader, epoch_num, device):
    model.train()
    total_loss = 0

    initial_state_dict = model.state_dict()
    
    print(f"--- Epoch {epoch_num} Train ---")
    data_loader = iter(data_loader) # TEST
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

            optimizer.zero_grad()
            
            # calculate masks
            src_pad_mask = model.create_pad_mask(X, pad_token=0).to(device)
            y_input_length = y_input.shape[1]
            tgt_mask = model.get_tgt_mask(y_input_length).to(device)

            # print("X: ", X)
            # print("y_input: ", y_input)
            # print("tgt_mask: ", tgt_mask)
            # print("src_pad_mask: ", src_pad_mask)
            pred = model(X, y_input, tgt_mask, src_pad_mask)
            
            pred = pred[:, -1, :] # only take the last predicted value
            loss = criterion(pred, y_expected)
            # print("y_expected: ", y_expected[0]) # TEST
            # print("argmax: ", torch.argmax(pred, dim=-1, keepdim=True)[0]) # TEST

            loss.backward()
            optimizer.step()

            loss_item = loss.detach().item()
            total_loss += loss_item

            if batch_num % 1000 == 0 and output_idx == 1:
                print(f"Batch {batch_num}, Loss: {loss_item}")

        
        if batch_num == 1: # TEST
            break
    
    average_loss = total_loss / 2 # len(data_loader) TEST
    print(f"--- Epoch {epoch_num} - Average Loss: {average_loss} ---")
    
    # current_state_dict = model.state_dict()
    # weights_same = all(torch.equal(initial_state_dict[key], current_state_dict[key]) for key in initial_state_dict.keys())
    # print(f"Weights same? {weights_same}")
    return average_loss

def evaluateModel(model, data_loader, device):
    model.eval()
    total_bleu = 0
    
    avg_pred_length = []
    outputs = []

    print(f"--- Epoch {epoch_num} Eval ---")
    data_loader = iter(data_loader) # TEST
    with torch.no_grad():
        for batch_number, batch in enumerate(data_loader):
            input_seq, target_seq = batch[0].to(device), batch[1].to(device)
            X, y = input_seq, target_seq

            y_input = y[:, :1]
            for output_idx in range(1, y.shape[1]):
                # skip if all values to be predicted are padded values
                if torch.all(y_input[:, -1] == 3):
                    for row_idx in range(y_input.shape[0]):
                        print("y_input Sequence: ", targetTokenizer.idx_to_seq(y_input[row_idx].tolist(), includeInvis=True, splitSpace=False))
                        outputs.append(([y[row_idx].tolist()], y_input[row_idx].tolist()))
                    break

                # filter all sequences already filtered (generated EOS)
                unfinished_filter = y_input[:, -1] != 3 # 3 is [EOS] token number for my tokenizers
                finished_filter = ~unfinished_filter
                y_finished = y_input[finished_filter]
                for row_idx in range(y_finished.shape[0]):
                    print("y_finished Sequence: ", targetTokenizer.idx_to_seq(y_finished[row_idx].tolist(), includeInvis=True, splitSpace=False))
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
            
            if batch_number == 1: # TEST
                break
    
    for true_label_list, generated in outputs:
        try:
            total_bleu += bleu.sentence_bleu(true_label_list, generated, weights=(0.25, 0.5, 0.25, 0))
        except:
            total_bleu += 0
    
    return total_bleu / len(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--on_pace", help="Whether it's running on a Pace cluster", action="store_true")
    parser.add_argument("-m", "--model_name", type=str, nargs='?', help="The model name")
    args = parser.parse_args()
    if args.on_pace:
        location_prefix = "/storage/home/hcoda1/9/ahoerler3/scratch/p-ahoerler3-1/MeeseeksPositionEncodings/"
        print("Intended for Pace")
    else:
        location_prefix = ""
    
    print("English as source, Chinese as target")

    model_path = f"{location_prefix}{args.model_name}.pth"
    train_data_loader, eval_data_loader = createDataloaders(on_pace=args.on_pace)
    in_vocab_size = len(seqTokenizer)
    out_vocab_size = len(targetTokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    
    model = TransformerModel(input_vocab_size=in_vocab_size, output_vocab_size=out_vocab_size).to(device)
    if os.path.exists(model_path):
        print("Loading model from file")
        model.load_state_dict(torch.load(model_path))
    
    criterion = torch.nn.CrossEntropyLoss()
    learning_rates = [0.0001, 0.00001, 0.000001]
    lr_iterator = iter(learning_rates)
    optimizer = torch.optim.Adam(model.parameters(), lr=next(lr_iterator))
    epochs = 40

    print("\nStarting Training/Testing")
    prev_loss = trainModel(model, criterion, optimizer, train_data_loader, 0, device)
    for epoch_num in range(1, epochs):
        loss = trainModel(model, criterion, optimizer, train_data_loader, epoch_num, device)
        torch.save(model.state_dict(), model_path)
        if loss < prev_loss / 2:
            next_lr = next(lr_iterator, None)
            if next_lr is not None:
                optimizer = torch.optim.Adam(model.parameters(), lr=next_lr)
                prev_loss = loss
        average_bleu = evaluateModel(model, train_data_loader, device)
        print(f"Epoch {epoch_num} - Average Bleu: {average_bleu}")