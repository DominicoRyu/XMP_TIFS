import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, embed_dim, cnn_size, rnn_size, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2*rnn_size+embed_dim, out_channels=cnn_size, kernel_size=kernel_size)
        self.LReLU = nn.LeakyReLU(0.3)
        self.maxpool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(in_channels=cnn_size, out_channels=cnn_size, kernel_size=kernel_size)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LReLU(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x

class ByteRCNNModel(nn.Module):
    def __init__(self, maxlen, embed_dim, rnn_size, cnn_size, kernels, output_cnt):
        super(ByteRCNNModel, self).__init__()
        self.embedding = nn.Embedding(maxlen, embed_dim)
        
        self.gru1 = nn.GRU(embed_dim, rnn_size, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2*rnn_size, rnn_size, batch_first=True, bidirectional=True)
        
        self.conv9 = ConvBlock(embed_dim=embed_dim, rnn_size=rnn_size, cnn_size=cnn_size, kernel_size=9)
        self.conv27 = ConvBlock(embed_dim=embed_dim, rnn_size=rnn_size, cnn_size=cnn_size, kernel_size=27)
        self.conv40 = ConvBlock(embed_dim=embed_dim, rnn_size=rnn_size, cnn_size=cnn_size, kernel_size=40)
        self.conv65 = ConvBlock(embed_dim=embed_dim, rnn_size=rnn_size, cnn_size=cnn_size, kernel_size=65)

        self.convs = nn.ModuleList([
                self.conv9,
                self.conv27,
                self.conv40,
                self.conv65,
        ])

        self.fc1 = nn.Linear(len(kernels)*cnn_size*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_cnt)

        self.init_weights()

    def init_weights(self):
        # weights
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

        for name, param in self.gru1.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        for name, param in self.gru2.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        # Bias
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)

        for name, param in self.gru1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
        for name, param in self.gru2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)

        print('Parameter initialization Complete')

    def forward(self, x):
        emb = self.embedding(x)
        emb = F.dropout(emb, 0.1)
        x_context, _ = self.gru1(emb)
        x_context, _ = self.gru2(x_context)
        x = torch.cat((emb, x_context), 2)
        
        x = x.permute(0, 2, 1)  # Conv1D에 맞게 차원 변경
        convs_out = [conv(x) for conv in self.convs]
        poolings_max = [F.adaptive_max_pool1d(conv, 1).view(conv.size(0), -1) for conv in convs_out]
        poolings_avg = [F.adaptive_avg_pool1d(conv, 1).view(conv.size(0), -1) for conv in convs_out]
        x = torch.cat(poolings_max + poolings_avg, 1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        outputs = self.output_layer(x)

        return outputs

if __name__=='__main__':
    net = ByteRCNNModel(maxlen=512, embed_dim=16, 
                    rnn_size=64, cnn_size=64, 
                    kernels=[9,27,40,65], output_cnt=75)
    print(net)
    