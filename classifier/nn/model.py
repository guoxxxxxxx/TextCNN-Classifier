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
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, config['lstm_hidden_layer'])),
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


class LSTMModelBlock(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModelBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.lstm(x)
        return x.unsqueeze(1)


class TextCNNBlock(nn.Module):

    def __init__(self):
        super(TextCNNBlock, self).__init__()
        # 特征提取层
        self.b0 = CNNBlock(5)
        self.b1 = CNNBlock(7)
        self.b2 = CNNBlock(11)
        self.b3 = CNNBlock(17)
        self.b4 = CNNBlock(25)
        self.b5 = CNNBlock(3)

        # 特征融合部分 及 输出头
        self.neck = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6 * config['hidden_layer'], 512)
        )

    def forward(self, x, label=None):

        b0_results = self.b0(x)
        b1_results = self.b1(x)
        b2_results = self.b2(x)
        b3_results = self.b3(x)
        b4_results = self.b4(x)
        b5_results = self.b5(x)

        features = torch.cat([b0_results, b1_results, b2_results, b3_results, b4_results,
                              b5_results], dim=1)
        output = self.neck(features)
        return output


class TextCNNModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(TextCNNModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix
        self.textCNN = TextCNNBlock()
        self.head = nn.Sequential(
            nn.Linear(512, config['nc'])
        )

    def forward(self, x):
        x = self.embedding_matrix(x)
        x = self.textCNN(x)
        x = self.head(x)
        return x


class LSTM_TextCNNModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM_TextCNNModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.embedding_matrix = embedding_matrix

        # 长短记忆神经网络先处理位置关系
        self.lstm = LSTMModelBlock(config['embedding_dim'], config['lstm_hidden_layer'])
        # 通过CNN再提取特征
        self.textCNN = TextCNNBlock()
        # 通过全连接层输出31个类别的概率
        self.head = nn.Linear(512, config['nc'])


    def forward(self, x):
        x = self.embedding_matrix(x)
        lstm_out = self.lstm(x)
        cnn_out = self.textCNN(lstm_out)
        output = self.head(cnn_out)
        return output
