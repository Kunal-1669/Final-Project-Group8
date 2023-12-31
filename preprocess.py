import nltk
import concurrent.futures
from nltk.tokenize import word_tokenize
from RegexRemover import get_cleaned_data
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import nltk
def tokenize_column(column):
    print("1")
    return column.apply(word_tokenize)


def multiprocess_tokenize(df, col_name):
    print("2")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for idx, grp in df.groupby(df.index // 1000):
            part_df = grp[col_name]
            futures.append(executor.submit(tokenize_column, part_df))

        tokenized_parts = []
        for f in concurrent.futures.as_completed(futures):
            tokenized_parts.append(f.result())

    return pd.concat(tokenized_parts)


def remove_stopwords(tokens):
    # print("3")
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

# Function for stemming
# def stem_tokens(tokens):
#     print("4")
#     ps = PorterStemmer()
#     return [ps.stem(token) for token in tokens]

from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# def lemmatize_words(tokens):
#     return ' '.join(lemmatizer.lemmatize(word) for word in tokens.split())
def lemmatize_words(tokens):
    # print("5")
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def apply_preprocessing(df, text_cols):
    for col in text_cols:
        df[col] = multiprocess_tokenize(df,col)
        df[col] = df[col].apply(remove_stopwords)
        # df[col] = df[col].apply(stem_tokens)
        df[col] = df[col].apply(lemmatize_words)
    return df


def train_word2vec_model(tokens_list):
    print('6')
    model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Embedding text with Word2Vec
# def embed_with_word2vec(model, df, cols):
#     # print('7')
#     for col in cols:
#         df[col] = df[col].apply(lambda x: [model.wv[word] for word in x if word in model.wv])
#     return df
def embed_with_word2vec(model, tokens_list):
    return [model.wv[word] for word in tokens_list if word in model.wv]


def df_clean():
    print("test")
    cleaned_train_data, cleaned_test_data = get_cleaned_data()
    print("regex done")

    text_cols = ['document', 'summary']
    num_samples = 1000

    # Sample the data
    sampled_train_data = cleaned_train_data.sample(n=num_samples, random_state=42)
    sampled_train_data.reset_index(drop=True, inplace=True)
    # Apply preprocessing
    cleaned_train_data = apply_preprocessing(sampled_train_data, text_cols)
    cleaned_test_data = apply_preprocessing(cleaned_test_data,text_cols)
    # cleaned_validation_data = apply_preprocessing(cleaned_validation_data,text_cols)

    # print(cleaned_train_data.head())
    # print(cleaned_test_data.head())
    # # print(cleaned_validation_data.head())

    # Train Word2Vec model on the training data
    cleaned_text = cleaned_train_data['document'].tolist() + cleaned_train_data['summary'].tolist()
    word2vec_model = train_word2vec_model(cleaned_text)

    # Embed train, test, and validation data with the trained Word2Vec model
    cleaned_train_data['document_embedding'] = cleaned_train_data['document'].apply(
        lambda x: embed_with_word2vec(word2vec_model, x))
    cleaned_train_data['summary_embedding'] = cleaned_train_data['summary'].apply(
        lambda x: embed_with_word2vec(word2vec_model, x))
    # df_cleaned_embedded_val = embed_with_word2vec(word2vec_model, cleaned_validation_data, text_cols)
    # cleaned_test_data['document_embedding'] = cleaned_test_data['document'].apply(embed_with_word2vec(word2vec_model, cleaned_train_data, text_cols))
    # cleaned_test_data['summary_embedding'] = cleaned_test_data['summary'].apply(embed_with_word2vec(word2vec_model, cleaned_test_data, text_cols))
    cleaned_test_data['document_embedding'] = cleaned_test_data['document'].apply(
        lambda x: embed_with_word2vec(word2vec_model, x))
    cleaned_test_data['summary_embedding'] = cleaned_test_data['summary'].apply(
        lambda x: embed_with_word2vec(word2vec_model, x))
    cleaned_train_data['document_embedding'] = cleaned_train_data['document_embedding'].apply(lambda x: np.array(x).astype(float))
    cleaned_train_data['summary_embedding'] = cleaned_train_data['summary_embedding'].apply(lambda x: np.array(x).astype(float))

    # cleaned_test_data['document_embedding'] = cleaned_test_data['document_embedding'].apply(lambda x: np.array(x).astype(float))
    # cleaned_test_data['summary_embedding'] = cleaned_test_data['summary_embedding'].apply(lambda x: np.array(x).astype(float))


    print('word2vec done')

    return cleaned_train_data, cleaned_test_data


# if __name__ == "__main__":
#     df_cleaned_embedded_train, df_cleaned_embedded_test = main()
df_train,df_test=df_clean()
df_train.to_csv('cleaned_train_data.csv', index=False)
df_test.to_csv('cleaned_test_data.csv', index=False)