import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
        Convolutional Neural Network (CNN) model for tabular data classification tasks.

        This model consists of two convolutional layers followed by max pooling operations,
        and two fully connected layers for classification.

        :param output_dim(int): Dimension of the output (number of classes).

        Attributes:
            conv1 (nn.Conv1d): First convolutional layer with 16 output channels and kernel size 3.
            maxpool1 (nn.MaxPool1d): Max pooling layer with kernel size 2.
            conv2 (nn.Conv1d): Second convolutional layer with 32 output channels and kernel size 3.
            relu (nn.ReLU): ReLU activation function.
            fc1 (nn.Linear): First fully connected layer with input size 224 and output size 128.
            fc2 (nn.Linear): Second fully connected layer with input size 128 and output size specified by output_dim.

        """
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)

    
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        output, (hn, cn) = self.lstm(x.unsqueeze(1), (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = F.relu(hn)
        out = self.fc(out)
        return F.sigmoid(out)
class CNN_LSTM(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return F.sigmoid(out)