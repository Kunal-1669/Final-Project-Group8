from datasets import load_dataset

class MultiNewsDataset:
    def __init__(self):
        self.dataset = load_dataset("multi_news")
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]
        self.validation_dataset = self.dataset["validation"]

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

