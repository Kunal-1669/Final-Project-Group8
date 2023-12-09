import random
# from preprocess import main
import preprocess
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Check for GPU availability
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print("GPU is available" if is_cuda else "GPU not available, CPU used")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


# Define the Seq2Seq model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pad_idx):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_idx = pad_idx
        self.dropout = nn.Dropout(dropout)

        # Using LSTM directly on embeddings
        self.encoder = nn.LSTM(hidden_size, hidden_size)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

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
        return torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(
            device)


def create_vocabulary(texts):
    # Create a vocabulary using the provided texts
    all_words = [word for text in texts for word in text.split()]
    word_counts = Counter(all_words)
    vocab = list(word_counts.keys())
    special_tokens = ['<pad>', '<start>', '<end>']
    vocab = special_tokens + vocab
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return word_to_index, index_to_word


# Assuming df_cleaned_embedded is the DataFrame obtained from your preprocessing
# Extract relevant columns for your Seq2Seq model (e.g., document and summary columns)

# df_cleaned_embedded_train, df_cleaned_embedded_test = main()

df_cleaned_embedded_test = preprocess.df_clean()
df_cleaned_embedded_train, _ = preprocess.df_clean()
df_cleaned_embedded_train = df_cleaned_embedded_train.sample(n=1000, random_state=42)
df_cleaned_embedded_train.reset_index(drop=True, inplace=True)

# Convert embedded data to tensors
def get_tensors(df):
    # Assuming 'document_vectors' and 'summary_vectors' are the columns with embeddings
    documents = torch.tensor(np.stack(df['document'].values))
    summaries = torch.tensor(np.stack(df['summary'].values))
    return documents, summaries


train_documents, train_summaries = get_tensors(df_cleaned_embedded_train)
# val_documents, val_summaries = get_tensors(df_cleaned_embedded_val)
test_documents, test_summaries = get_tensors(df_cleaned_embedded_test)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_documents, train_summaries)
# val_dataset = TensorDataset(val_documents, val_summaries)
test_dataset = TensorDataset(test_documents, test_summaries)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Model Initialization
hidden_size = 100  # Assuming the Word2Vec vector size is 100
output_size = hidden_size  # Output size is same as input for Seq2Seq
pad_idx = 0  # Assuming 0 is the padding index in embeddings

model = Seq2SeqLSTM(hidden_size, output_size, pad_idx, dropout=0.5)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_src, batch_trg in train_dataloader:
        batch_src, batch_trg = batch_src.to(device), batch_trg.to(device)
        optimizer.zero_grad()
        output = model(batch_src, batch_trg)
        output_dim = output.shape[-1]

        # Reshape output and target for computing loss
        output = output[:, 1:].reshape(-1, output_dim)
        trg = batch_trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} - Training Loss: {avg_train_loss}")

    # # Validation loop
    # model.eval()
    # total_val_loss = 0
    # with torch.no_grad():
    #     for batch_src, batch_trg in val_dataloader:
    #         batch_src, batch_trg = batch_src.to(device), batch_trg.to(device)
    #         output = model(batch_src, batch_trg, 0)  # No teacher forcing in validation
    #         output_dim = output.shape[-1]
    #
    #         output = output[:, 1:].reshape(-1, output_dim)
    #         trg = batch_trg[:, 1:].reshape(-1)
    #
    #         loss = criterion(output, trg)
    #         total_val_loss += loss.item()
    #
    # avg_val_loss = total_val_loss / len(val_dataloader)
    # print(f"Epoch {epoch} - Validation Loss: {avg_val_loss}")

# Test loop
model.eval()
total_test_loss = 0
with torch.no_grad():
    for batch_src, batch_trg in test_dataloader:
        batch_src, batch_trg = batch_src.to(device), batch_trg.to(device)
        output = model(batch_src, batch_trg, 0)  # No teacher forcing in test
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        trg = batch_trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Test Loss: {avg_test_loss}")


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_src, batch_trg in dataloader:
            batch_src, batch_trg = batch_src.to(device), batch_trg.to(device)
            output = model(batch_src, batch_trg, 0)  # No teacher forcing in evaluation
            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = batch_trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = output.argmax(dim=1, keepdim=True)
            correct_predictions += predictions.eq(trg.view_as(predictions)).sum().item()
            total_predictions += trg.shape[0]

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


# Evaluate on test data
test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Function to load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()


# Example Usage
save_model(model, 'seq2seq_model.pth')
# To load: load_model(model, 'seq2seq_model.pth')
