# import concurrent.futures
# from nltk.tokenize import word_tokenize
# from RegexRemover import get_cleaned_data
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# def tokenize_column(column):
#     return column.apply(word_tokenize)
#
#
# def multiprocess_tokenize(df, col_name):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = []
#         for idx, grp in df.groupby(df.index // 1000):
#             part_df = grp[col_name]
#             futures.append(executor.submit(tokenize_column, part_df))
#
#         tokenized_parts = []
#         for f in concurrent.futures.as_completed(futures):
#             tokenized_parts.append(f.result())
#
#     return pd.concat(tokenized_parts)
#
#
#
# def remove_stopwords(tokens):
#     stop_words = set(stopwords.words('english'))
#     return [token for token in tokens if token.lower() not in stop_words]
#
# # Function for stemming
# def stem_tokens(tokens):
#     ps = PorterStemmer()
#     return [ps.stem(token) for token in tokens]
# def main():
#     cleaned_train_data, cleaned_test_data = get_cleaned_data()
#
#     text_cols = ['document', 'summary']
#     for col in text_cols:
#         cleaned_train_data[col] = multiprocess_tokenize(cleaned_train_data, col)
#     print(cleaned_train_data)
# if __name__ == "__main__":


import concurrent.futures
from nltk.tokenize import word_tokenize
from RegexRemover import get_cleaned_data
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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


def apply_preprocessing(df, text_cols):
    for col in text_cols:
        df[col] = df[col].apply(remove_stopwords)
        df[col] = df[col].apply(stem_tokens)
    return df


def main():
    cleaned_train_data, cleaned_test_data = get_cleaned_data()

    text_cols = ['document', 'summary']

    # Tokenize
    for col in text_cols:
        cleaned_train_data[col] = multiprocess_tokenize(cleaned_train_data, col)

    # Apply stopwords removal and stemming
    cleaned_train_data = apply_preprocessing(cleaned_train_data, text_cols)

    print(cleaned_train_data)


if __name__ == "__main__":
    main()

#     main()