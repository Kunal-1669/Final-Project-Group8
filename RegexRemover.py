import re
from dataload import get_train_and_test_dataframes
import warnings
from concurrent.futures import ProcessPoolExecutor

# Ignore the FutureWarning related to datasets
warnings.filterwarnings("ignore", category=FutureWarning, module="datasets.table")

class TextStripper:
    def __init__(self):
        pass

    def text_strip(self, text):
        cleaned_text = re.sub("[\t\r\n]+", ' ', str(text)).lower()  # remove escape characters
        cleaned_text = re.sub("(__+|-+|~+|\++)", ' ', cleaned_text).lower()  # remove repeating _, -, ~, or + characters
        cleaned_text = re.sub("(\.\.+)", ' ', cleaned_text).lower()  # remove repeating . characters
        cleaned_text = re.sub("\s+", ' ', cleaned_text).lower()  # remove multiple spaces
        cleaned_text = re.sub("\s.\s", ' ', cleaned_text).lower()  # remove any single characters hanging between 2 spaces
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text).lower() # remove punctuation
        cleaned_text = re.sub(r'http\S+', '',cleaned_text).lower() # remove url
        cleaned_text = re.sub(r'[{}()\[\]]', '', cleaned_text).lower()  # remove all types of braces

        return cleaned_text

    def clean_columns(self, df, columns):
        with ProcessPoolExecutor() as executor:
            for column_name in columns:
                df[column_name] = list(executor.map(self.text_strip, df[column_name]))
        return df

def get_cleaned_data():
    # Load the train and test DataFrames
    df_train, df_test = get_train_and_test_dataframes()

    text_stripper = TextStripper()

    # Specify the columns you want to clean
    text_column_names = ['document', 'summary']  # Change these to the actual column names you want to clean

    # Clean the text columns in both train and test DataFrames
    df_train_cleaned = text_stripper.clean_columns(df_train.copy(), text_column_names)
    df_test_cleaned = text_stripper.clean_columns(df_test.copy(), text_column_names)

    return df_train_cleaned, df_test_cleaned

if __name__ == "__main__":
    # Get the cleaned DataFrames
    cleaned_train_data, cleaned_test_data = get_cleaned_data()

    # Print the cleaned DataFrames
    print("Cleaned Train Dataset:")
    print(cleaned_train_data.head())

    print("\nCleaned Test Dataset:")
    print(cleaned_test_data.head())
