import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset(Dataset):
    def __init__(self, data: npt.NDArray,
                 prop_missing: Optional = None):
        """
        Initialize Dataset object

        Args:
        data (numpy.ndarray): Array containing data.
        prop_missing (float): Proportion of missing data to simulate.
        """
        self.data = data

        if prop_missing is not None:
            # Create a mask to simulate missing data
            self.input_mask = np.random.rand(*self.data.shape) > prop_missing
            self.data_missing = np.where(self.input_mask, self.data, 0.0)
        else:
            self.input_mask = np.ones_like(self.data, dtype=bool)
            self.data_missing = self.data.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
        idx (int): Index of the item to retrieve.

        Returns:
        Tuple: A tuple containing the missing data, the complete data, and the input mask.
        """
        return self.data_missing[idx], self.data[idx], self.input_mask[idx], self.input_mask[idx].astype(int)

    def get_missing_rate(self):
        print(f'Missing rate: {np.round(np.mean(self.input_mask == 0), 2)}')


class DataModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for handling data loading and preprocessing.

    Args:
        dataset (str): The name of the dataset to load. Must be either 'credit' or 'spam'.
        batch_size (int): The size of each batch to load.
        normalize (bool): Whether to normalize the data.
        val_len (float): The proportion of the data to use for validation.
        test_len (float): The proportion of the data to use for testing.
        prop_missing (float): The proportion of values in the data to set to NaN to simulate missing data.
    """
    def __init__(self,
                 dataset: str = 'credit',
                 batch_size: int = 128,
                 normalize: bool = False,
                 val_len: float = 0.1,
                 test_len: float = 0.1,
                 prop_missing: float = 0.2):

        super().__init__()

        import os
        # Load the data from a CSV file based on the specified dataset name
        if dataset == 'credit':
            self.data = pd.read_csv('./data/credit.csv')
            self.data = self.data.drop(columns=['default.payment.next.month', 'ID'])
        elif dataset == 'cancer':
            self.data = pd.read_csv('./data/breast.csv')
            self.data = self.data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
        elif dataset == 'news':
            self.data = pd.read_csv('./data/news.csv')
            self.data = self.data.drop(columns=['url', ' timedelta', ' shares'])
        elif dataset == 'spam':
            self.data = pd.read_csv('./data/spam.csv')
        elif dataset == 'letter':
            self.data = pd.read_csv('./data/letter.csv')

        # Normalize the data if requested
        if normalize:
            self.normalizer = MinMaxScaler()
            self.data = pd.DataFrame(self.normalizer.fit_transform(self.data), columns=self.data.columns)

        # Convert the data to a numpy array
        self.data_numpy = self.data.to_numpy().astype(np.float32)

        self.prop_missing = prop_missing
        self.batch_size = batch_size
        self.val_len = val_len
        self.test_len = test_len
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage=None):
        """
        Splits the data into train, validation, and test sets, and creates PyTorch DataLoader objects for each set.

        Args:
            stage: (str): The stage of training (fit or test). Unused in this implementation.
        """

        # Split the data into train, validation, and test sets using train_test_split
        train, test = train_test_split(self.data_numpy, test_size=self.val_len + self.test_len)
        val, test = train_test_split(test, test_size=self.val_len / (self.val_len + self.test_len))

        # Create Dataset objects for each set with missing values introduced according to prop_missing
        train = Dataset(train, prop_missing=self.prop_missing)
        val = Dataset(val, prop_missing=self.prop_missing)
        test = Dataset(test, prop_missing=self.prop_missing)

        # Create DataLoader objects for each set
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)

    def input_size(self):
        return self.data.shape[1]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
