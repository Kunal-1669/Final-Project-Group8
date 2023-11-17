import re
from dataload import get_train_and_test_dataframes
import warnings
from concurrent.futures import ProcessPoolExecutor
# Ignore the FutureWarning related to datasets
warnings.filterwarnings("ignore", category=FutureWarning, module="datasets.table")
df_train, df_test = get_train_and_test_dataframes()
class TextStripper:
    def __init__(self):
        pass

    def text_strip(self, column):
        cleaned_text = ''
        for row in column:           # Combine similar replacements
            row = re.sub("[\t\r\n]+", ' ', str(row)).lower()  # remove escape characters

            row = re.sub("(__+|-+|~+|\++)", ' ', str(row)).lower()  # remove repeating _, -, ~, or + characters
            row = re.sub("(\.\.+)", ' ', str(row)).lower()  # remove repeating . characters

            row = re.sub("\s+", ' ', str(row)).lower()  # remove multiple spaces

            # Should always be last
            row = re.sub("\s.\s", ' ', str(row)).lower()  # remove any single characters hanging between 2 spaces

            cleaned_text += row

        return cleaned_text

text_stripper = TextStripper()

# Specify the column you want to clean
text_column_name = 'document'  # Change this to the actual column name you want to clean

with ProcessPoolExecutor() as executor:
    # Apply text_strip method to the specified column in df_train
    df_train[text_column_name] = list(executor.map(text_stripper.text_strip, df_train[text_column_name]))

    # Apply text_strip method to the specified column in df_test
    df_test[text_column_name] = list(executor.map(text_stripper.text_strip, df_test[text_column_name]))



# Print the cleaned DataFrames
print("Cleaned Train Dataset:")
print(df_train.head())

print("\nCleaned Test Dataset:")
print(df_test.head())