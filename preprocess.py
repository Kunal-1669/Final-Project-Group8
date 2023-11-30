
import concurrent.futures
from nltk.tokenize import word_tokenize
from RegexRemover import get_cleaned_data
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec


def tokenize_column(column):
    return column.apply(word_tokenize)


def multiprocess_tokenize(df, col_name):
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
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

# Function for stemming
def stem_tokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(tokens):
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens.split())


def apply_preprocessing(df, text_cols):
    for col in text_cols:
        df[col] = df[col].apply(remove_stopwords)
        df[col] = df[col].apply(stem_tokens)
    return df



def train_word2vec_model(tokens_list):
    model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    return model


def main():
    cleaned_train_data, cleaned_test_data = get_cleaned_data()

    text_cols = ['document', 'summary']

    # Tokenize
    for col in text_cols:
        cleaned_train_data[col] = multiprocess_tokenize(cleaned_train_data, col)

    # Apply stopwords removal and stemming
    cleaned_train_data = apply_preprocessing(cleaned_train_data, text_cols)

    print(cleaned_train_data)
    #
    # cleaned_text = (cleaned_train_data['document'] + ' ' + cleaned_train_data['summary']).tolist()
    # model = train_word2vec_model(cleaned_text)
    cleaned_doc_text = cleaned_train_data['document'].tolist()
    cleaned_summary_text = cleaned_train_data['summary'].tolist()

    cleaned_text = cleaned_doc_text + cleaned_summary_text
    model = train_word2vec_model(cleaned_text)
    word_embeddings = model.wv

    df_doc_embeddings = pd.DataFrame(cleaned_train_data['document'].apply(lambda x: [word_embeddings[word] for word in x]))
    df_summary_embeddings = pd.DataFrame(cleaned_train_data['summary'].apply(lambda x: [word_embeddings[word] for word in x]))
    df_all_embeddings = pd.concat([df_doc_embeddings, df_summary_embeddings], axis=1)
    # print(df_all_embeddings)
    # print(cleaned_train_data)
    df_cleaned_embedded = pd.concat([cleaned_train_data, df_all_embeddings],axis=1)
    return df_cleaned_embedded


if __name__ == "__main__":
    main()

#     main()