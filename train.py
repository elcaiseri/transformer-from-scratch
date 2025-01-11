import torch
from torch.utils.data import DataLoader
from dataset import Tokenizer, TranslationDataset
from model import Transformer
from inference import translate_sentence
from utils import df_column_to_vocab
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

import pandas as pd

def train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, device, epoch, num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for i, batch in progress_bar:
        src_input_ids, tgt_input_ids = batch["src_input_ids"].to(device), batch["tgt_input_ids"].to(device)
        tgt_input, tgt_output = tgt_input_ids[:, :-1], tgt_input_ids[:, 1:]

        src_mask = (src_input_ids != SRC_PAD_IDX).long().unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.tril(torch.ones(tgt_input.size(1), tgt_input.size(1))).unsqueeze(0).to(device)

        optimizer.zero_grad()

        logits = model(src_input_ids, tgt_input, src_mask, tgt_mask)  # (batch_size, tgt_len, tgt_vocab_size)
        logits = logits.view(-1, logits.size(-1))  # (batch_size * tgt_len, tgt_vocab_size)
        tgt_output = tgt_output.contiguous().view(-1)  # (batch_size * tgt_len)
        loss = criterion(logits, tgt_output)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/(i+1))

    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")
    
# Train the model
def train(model, loader, criterion, optimizer, lr_scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        train_one_epoch(model, loader, criterion, optimizer, lr_scheduler, device, epoch, num_epochs)
    # Save the model
    torch.save(model.state_dict(), "assets/transformer_model_epoch_{}.pth".format(epoch+1))
    print("Model saved successfully with special note: 'Training complete for epoch {}!'".format(epoch+1))
        
if __name__ == "__main__":
    
    # Hyperparameters
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 128
    num_epochs = 99
    learning_rate = 2e-5
    max_len = 64
    warmup_steps = 4000
    # print hyperparameters
    print("Hyperparameters:", num_layers, d_model, num_heads, d_ff, dropout, batch_size, num_epochs, learning_rate, max_len, warmup_steps)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    data = pd.read_csv('./data/sample_data.csv')

    en_vocab = df_column_to_vocab(data, 'english')
    de_vocab = df_column_to_vocab(data, 'german')

    en_tokenizer = Tokenizer(en_vocab, "assets/en_vocab.json")
    de_tokenizer = Tokenizer(de_vocab, "assets/de_vocab.json")

    dataset = TranslationDataset(data, en_tokenizer, de_tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    SRC_PAD_IDX = en_tokenizer.encode('<PAD>')[0]
    TGT_PAD_IDX = de_tokenizer.encode('<PAD>')[0]

    # Initialize model, loss function, and optimizer
    model = Transformer(en_tokenizer.vocab_size, de_tokenizer.vocab_size, num_layers, d_model, num_heads, d_ff, dropout).to(device)
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train the model
    train(model, loader, criterion, optimizer, lr_scheduler, device, num_epochs)