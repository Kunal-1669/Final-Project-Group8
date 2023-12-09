# import numpy as np
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import preprocess
# # Assuming df_cleaned_embedded_train is your preprocessed and embedded training data
# # Make sure your 'summary' column is the target variable, and 'document' is the input text
#
# # df_cleaned_embedded_train = preprocess.df_clean()
# df_cleaned_embedded_test = preprocess.df_clean()
# df_cleaned_embedded_train, _ = preprocess.df_clean()
# df_cleaned_embedded_train = df_cleaned_embedded_train.sample(n=1000, random_state=42)
# df_cleaned_embedded_train.reset_index(drop=True, inplace=True)
# # df_sample_train=
# # Prepare the data
# # X = df_cleaned_embedded_train['document'].apply(lambda x: [np.array(sentence) for sentence in x])
# # # X = [[word.decode('utf-8') for word in sentence] for sentence in X]
# # # input_tokenizer.fit_on_texts(X)
# # y = df_cleaned_embedded_train['summary'].apply(lambda x: [np.array(sentence) for sentence in x])
# X = df_cleaned_embedded_train['document'].tolist()
# y = df_cleaned_embedded_train['summary'].tolist()
# # print(X)
# # print(Y)
#
# # Tokenize input sequences
# input_tokenizer = Tokenizer()
# input_tokenizer.fit_on_texts(X)
# vocab_size_encoder = len(input_tokenizer.word_index) + 1
# max_len_encoder = max(len(sentence) for sentence in X)
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
# latent_dim = 100
# #
# # # Encoder
# # encoder_inputs = Input(shape=(max_len_encoder,latent_dim))
# # embedding_layer = Embedding(vocab_size_encoder, latent_dim, input_length=max_len_encoder, mask_zero=True)
# # encoder_embedding = embedding_layer(encoder_inputs)
# # encoder_lstm = LSTM(latent_dim, return_state=True)
# # encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
# # encoder_states = [state_h, state_c]
# # #
# # # # Decoder
# # # decoder_inputs = Input(shape=(max_len_decoder,))
# # # decoder_embedding = embedding_layer(decoder_inputs)
# # # decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# # # decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
# # # decoder_dense = Dense(vocab_size_decoder, activation='softmax')
# # # decoder_outputs = decoder_dense(decoder_outputs)
# # vocab_size_encoder=1000
# # vocab_size_decoder=1000
# # encoder_inputs = Input(shape=(max_len_encoder, latent_dim))
# # encoder_lstm = LSTM(latent_dim, return_state=True)
# # encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# #
# # # Decoder
# # decoder_inputs = Input(shape=(max_len_decoder, latent_dim))
# # decoder_lstm = LSTM(latent_dim, return_sequences=True)
# # decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
# # decoder_dense = Dense(vocab_size_decoder, activation='softmax')
# # decoder_outputs = decoder_dense(decoder_outputs)
# #
# # # Compile the model
# # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #
# # # Train the model
# # model.fit([X_train, y_train[:, :-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:], epochs=10, batch_size=64, validation_data=([X_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
# # Reshape your input data
# print("Original shape of X_train:", X_train.shape)
# print("Total size of X_train:", X_train.size)
# # X_train = X_train.reshape((X_train.shape[0], max_len_encoder, latent_dim))
# # X_val = X_val.reshape((X_val.shape[0], max_len_encoder, latent_dim))
# new_shape = (X_train.shape[0], X_train.shape[1] // latent_dim, latent_dim)
# X_train = X_train.reshape(new_shape)
#
# new_shape_val = (X_val.shape[0], X_val.shape[1] // latent_dim, latent_dim)
# X_val = X_val.reshape(new_shape_val)
# # Define the model
# encoder_inputs = Input(shape=(max_len_encoder, latent_dim))
# encoder_lstm = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
#
# # Decoder
# decoder_inputs = Input(shape=(max_len_decoder, latent_dim))
# decoder_lstm = LSTM(latent_dim, return_sequences=True)
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
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())