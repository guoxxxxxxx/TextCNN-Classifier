"""
模型结构
如果想要修改网络结构的话在这个文件夹修改即可
"""
import torch
from torch import nn

from classifier.conf.readConfig import Config

config = Config().config


class CNNBlock(nn.Module):
    def __init__(self, kernel_size, out_channels=config['hidden_layer']):
        super(CNNBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, config['embedding_dim'])),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=(config['max_length'] - kernel_size + 1))
        self.avepool = nn.AvgPool1d(kernel_size=(config['max_length'] - kernel_size + 1))

    def forward(self, bath_embedding):
        x = self.sequential(bath_embedding)
        x = x.squeeze(-1)   # 删掉最后一个维度
        x_maxpool = self.maxpool(x)
        x_maxpool = x_maxpool.squeeze(-1)  # 删掉最后一个维度
        # x_avepool = self.avepool(x)
        # x_avepool = x_avepool.squeeze(-1)
        # concat_x = torch.cat((x_maxpool, x_avepool), dim=1)
        # return concat_x
        return x_maxpool


class LSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)


    def forward(self, x):
        x = x.squeeze(1)
        x, (h_n, c_n) = self.lstm(x)
        return x, h_n, c_n


class TextCNNBlock(nn.Module):

    def __init__(self):
        super(TextCNNBlock, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 特征提取层
        self.block_list = [CNNBlock(ks).to(device=device) for ks in config['conv_kernel_list']]

    def forward(self, x, label=None):

        result = [block(x) for block in self.block_list]
        features = torch.cat(result, dim=1)

        return features


class TextCNNModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(TextCNNModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix
        self.textCNN = TextCNNBlock()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(len(config['conv_kernel_list']) * config['hidden_layer'], 512),
            nn.Linear(512, config['nc'])
        )

    def forward(self, x):
        x = self.embedding_matrix(x)
        x = self.textCNN(x)
        x = self.head(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTMModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix

        self.lstm = LSTMBlock(config['embedding_dim'], config['lstm_hidden_layer'],
                              config['lstm_num_layers'], bidirectional=config['lstm_bidirectional'])

        self.times = 2 if config['lstm_bidirectional'] else 1
        self.head = nn.Sequential(
            nn.Linear(config['lstm_hidden_layer'] * self.times * config['lstm_num_layers'], config['nc'])
        )

    def forward(self, x):
        x = self.embedding_matrix(x)
        _, x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)  # [num_layers * num_dire, b_s, h_s] => [b_s, n_l * n_d, h_s]
        encoding = x.reshape(x.shape[0], -1)
        x = self.head(encoding)
        return x


class LSTM_TextCNNModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM_TextCNNModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix

        self.textcnn = TextCNNBlock()


    def forward(self, x):
        x = self.embedding_matrix(x)
        x = self.textcnn(x)
