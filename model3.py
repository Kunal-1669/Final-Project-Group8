import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
df=pd.read_csv("cleaned_train_data.csv")
print(df.head())
print(df.shape)
X = df['document'].tolist()
y = df['summary'].tolist()
# print(X)
# print(Y)

# # Tokenize input sequences
# input_tokenizer = Tokenizer()
# input_tokenizer.fit_on_texts(X)
# vocab_size_encoder = len(input_tokenizer.word_index) + 1
# # max_len_encoder = max(len(sentence) for sentence in X)
# max_len_encoder=1244
# X = pad_sequences(input_tokenizer.texts_to_sequences(X), maxlen=max_len_encoder, padding='post')
#
# # Tokenize target sequences
# output_tokenizer = Tokenizer()
# output_tokenizer.fit_on_texts(y)
# vocab_size_decoder = len(output_tokenizer.word_index) + 1
# max_len_decoder = max(len(sentence) for sentence in y)
# y = pad_sequences(output_tokenizer.texts_to_sequences(y), maxlen=max_len_decoder, padding='post')
#
# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define the model
# latent_dim = 124
# # X_train = X_train.reshape((800, 1244, 124))
#
# print("Original shape of X_train:", X_train.shape)
# print("Total size of X_train:", X_train.size)
# # X_train = X_train.reshape((X_train.shape[0], max_len_encoder, latent_dim))
# # X_val = X_val.reshape((X_val.shape[0], max_len_encoder, latent_dim))
# # new_shape = (X_train.shape[0], X_train.shape[1] // latent_dim * latent_dim)
# # X_train = X_train.reshape(new_shape)
# # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
# # X_train = X_train.reshape((X_train.shape[0], max_len_encoder, latent_dim))
#
# new_shape_val = (X_val.shape[0], X_val.shape[1] // latent_dim * latent_dim)
# # X_val = X_val.reshape(new_shape_val)
# # X_val = X_val.reshape((X_val.shape[0], max_len_encoder,latent_dim))
# # Define the model
# encoder_inputs = Input(shape=(max_len_encoder, ))
# # encoder_lstm = LSTM(latent_dim, return_state=True, input_shape=(max_len_encoder,))
# encoder_lstm = LSTM(latent_dim, return_state=True,
#                    input_shape=(max_len_encoder,),
#                    return_sequences=False)
# # encoder_lstm = LSTM(latent_dim, return_state=True)
# # encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)encoder_inputs
# encoder_outputs = encoder_lstm(encoder_inputs)
# state_h, state_c = encoder_lstm.states
# # Decoder
# decoder_inputs = Input(shape=(max_len_decoder, latent_dim))
# # decoder_lstm = LSTM(latent_dim, return_sequences=True)
# # decoder_lstm = LSTM(latent_dim, return_sequences=True, input_shape=(max_len_decoder,))
# decoder_lstm = LSTM(latent_dim, return_sequences=True,
#                    input_shape=(max_len_decoder,),
#                    return_state=False)
# # decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
# decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
# decoder_dense = Dense(vocab_size_decoder, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
#
# # Compile the model
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit([X_train, y_train[:, :-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:], epochs=10, batch_size=64, validation_data=([X_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed

# Assuming your dataframe is named 'df'
df2 = df.iloc[:100]

input_sequences = df2['document'].apply(lambda x: ' '.join(x)).values
output_sequences = df2['summary'].apply(lambda x: ' '.join(x)).values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_sequences)
tokenizer.fit_on_texts(output_sequences)

input_sequences = tokenizer.texts_to_sequences(input_sequences)
output_sequences = tokenizer.texts_to_sequences(output_sequences)

max_len_input = max(len(seq) for seq in input_sequences)
max_len_output = max(len(seq) for seq in output_sequences)

# padded_input = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
# padded_output = pad_sequences(output_sequences, maxlen=max_len_output, padding='post')
max_len = max(max_len_input, max_len_output)

padded_input = pad_sequences(input_sequences, maxlen=max_len, padding='post')
padded_output = pad_sequences(output_sequences, maxlen=max_len, padding='post')


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_input, padded_output, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 100  # You may adjust this based on your data
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len_input))
model.add(LSTM(100, return_sequences=True))  # Return sequences for TimeDistributed layer
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Adjust patience as needed
# checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, callbacks=[early_stopping])
