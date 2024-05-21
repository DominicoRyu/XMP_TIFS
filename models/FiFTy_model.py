import torch.nn as nn

class FiFTy(nn.Module):
    """ Convolutional Neural Network architecture. """
    def __init__(self, layers, embed_size, filters, kernel, pool, dense1, dense2):
        super(FiFTy, self).__init__()
        self.embedding = nn.Embedding(256, embed_size)
        self.convs = nn.ModuleList([nn.Conv1d(embed_size if i == 0 else filters, filters, kernel) for i in range(layers)])
        self.pools = nn.ModuleList([nn.MaxPool1d(pool) for _ in range(layers)])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(filters, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.l_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        for conv, pool in zip(self.convs, self.pools):
            x = pool(self.l_relu(conv(x)))
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.l_relu(self.fc1(x))
        return self.fc2(x)