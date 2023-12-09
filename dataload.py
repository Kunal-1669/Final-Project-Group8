import data_download
import pandas as pd
# Create an instance of MultiNewsDataset
# data = data_download.MultiNewsDataset()

# # Access the train_dataset and test_dataset
# train_data = data.get_train_dataset()
# test_data = data.get_test_dataset()
#
# # Print some samples from the train dataset
# print("Train Dataset Sample:")
# for i in range(3):  # Print the first 3 samples for demonstration
#     print(train_data[i])
#     print("=" * 50)
#
# # Print some samples from the test dataset
# print("\nTest Dataset Sample:")
# for i in range(3):  # Print the first 3 samples for demonstration
#     print(test_data[i])
#     print("=" * 50)
# df_train = pd.DataFrame(train_data)
# df_test=pd.DataFrame(test_data)
# print(df_train)
# print(df_test)
#
# df_test=pd.DataFrame(test_data)

def get_train_and_test_dataframes():
    # Create an instance of MultiNewsDataset
    data = data_download.MultiNewsDataset()

    # Access the train_dataset and test_dataset
    train_data = data.get_train_dataset()
    test_data = data.get_test_dataset()
    # validation_data = data.get_validation_dataset()

    # Convert datasets to DataFrames
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    # df_validation = pd.DataFrame(validation_data)
    # df_train = df_train.sample(n=10000, random_state=42)
    # df_test = df_test.sample
    # print(type(df_train))
    # print(type(df_test))
    # return df_train_sampled, df_test_sampled

    return df_train, df_test
            # , df_validation
# print((get_train_and_test_dataframes()))
