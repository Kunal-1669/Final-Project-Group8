import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from collections import Counter
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

# Define the Seq2Seq model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pad_idx):
        super(Seq2SeqLSTM, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.pad_idx = pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        hidden, cell = self.init_hidden_cell(batch_size)

        embedded = self.embedding(src)
        encoder_output, (hidden, cell) = self.encoder(embedded)

        # Decoder input
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            prediction = self.fc(output)
            outputs[:, t] = prediction

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = trg[:, t] if teacher_force and t < trg_len else top1

        return outputs

    def init_hidden_cell(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(device)

# Hyperparameters
input_size = len(vocab) + 1  # Vocabulary size
hidden_size = 256
output_size = len(vocab) + 1  # Vocabulary size
pad_idx = 0

# Create model instance
model = Seq2SeqLSTM(input_size, hidden_size, output_size, pad_idx)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch_src, batch_trg in train_loader:
        optimizer.zero_grad()
        output = model(batch_src, batch_trg)
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        trg = batch_trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()


