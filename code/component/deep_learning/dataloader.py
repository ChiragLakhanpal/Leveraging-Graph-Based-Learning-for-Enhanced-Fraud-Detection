import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    """Custom dataset for tabular data."""
    def __init__(self, X, y):
        """
        :param X: preprocessed features
        :param y: target labels
                """
        self.X = X
        self.y = y

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """Retrieves a single data sample and its corresponding label at the given index."""
        return self.X[idx], self.y[idx]
    
    
class CustomDataLoader:
    """ Prepares custom dataloader for loading train, validation and test data sets"""
    def __init__(self, batch_size, device):
        """
        batch_size (int): The batch size to be used for loading data.
        device (torch.device): The device on which the data will be loaded (e.g., 'cpu' or 'cuda').
                        """
        self.batch_size = batch_size
        self.device = device

    def prepare_train_val_loader(self, X_train, y_train, X_val, y_val):
        """
        Prepares the data loaders for training and validation data sets.

        :param X_train: Input features for the training data.
        :param y_train: Target labels for the training data.
        :param X_val: Input features for the validation data.
        :param y_val: Target labels for the validation data.

        :returns train_loader: Data loader for the training data.
        :returns val_loader: Data loader for the validation data.
        :returns input_dim: The dimension of the input features.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train.values, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y_val.values, dtype=torch.float32, device=self.device)

        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)

        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, X_train.shape[1]

    def prepare_test_loader(self, X_test, y_test):
        """
        Prepares the data loader for the test data set.

        :param X_test: Input features for the test data.
        :param y_test: Target labels for the test data.

        :returns test_loader: Data loader for the test data.
        :returns input_dim: The dimension of the input features.
        """
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)

        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test.values, dtype=torch.float32, device=self.device)

        X_test = X_test.unsqueeze(1)

        test_dataset = CustomDataset(X_test, y_test)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return test_loader, X_test.shape[1]
    