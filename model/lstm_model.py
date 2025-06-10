import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, maxlen):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.lstm1 = nn.LSTM(embed_size, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128 * 2, 64, batch_first=True, bidirectional=True)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(64 * 2, 1024)
        self.dropout1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.25)
        
        self.fc5 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(0.25)
        
        self.fc6 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.embedding(x)                      # (batch_size, maxlen, embed_size)
        
        x, _ = self.lstm1(x)                       # (batch_size, maxlen, 256)
        x, _ = self.lstm2(x)                       # (batch_size, maxlen, 128)
        
        x = x.permute(0, 2, 1)                     # (batch_size, 128, maxlen)
        x = self.global_max_pool(x).squeeze(2)     # (batch_size, 128)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        
        x = self.fc6(x)                            # (batch_size, 4)
        return F.softmax(x, dim=1)                 # Apply softmax over classes

# model = TextClassificationModel(vocab_size=10000, embed_size=300, maxlen=100)


