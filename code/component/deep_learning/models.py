import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, in_size,cnn_hidden_size,hidden_size,cnn_out_size,output_dim,kernel_size,kernel_size_pool):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_size, out_channels=cnn_hidden_size, kernel_size=kernel_size)
        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size_pool)
        self.conv2 = nn.Conv1d(in_channels=cnn_hidden_size, out_channels=cnn_out_size, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

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
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = F.relu(hn)
        out = self.fc(out)
        return F.sigmoid(out)

class CNN_LSTM(nn.Module):
    def __init__(self, in_size,hidden_size,hidden_size_2,num_layers,cnn_out,output_size,kernel_size,kernel_size_pool,stride,padding):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_size, out_channels=cnn_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_pool, stride=stride),
            nn.Conv1d(in_channels=cnn_out, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_pool, stride=stride)
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size_2, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return F.sigmoid(out)