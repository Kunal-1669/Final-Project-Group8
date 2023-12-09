import torch
from torch.utils.data import TensorDataset, DataLoader
import preprocess

df_cleaned_embedded_train = preprocess.df_clean()
df_cleaned_embedded_test = preprocess.df_clean()
print(df_cleaned_embedded_train.head())

# Convert embedded data to tensors
def get_tensors(df):
    # Assuming 'document_vectors' and 'summary_vectors' are the columns with embeddings
    documents = torch.tensor(np.stack(df['document_embedded'].values))
    summaries = torch.tensor(np.stack(df['summary_embedded'].values))
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
