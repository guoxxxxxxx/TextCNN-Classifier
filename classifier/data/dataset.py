import json

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import logging

from classifier.conf.readConfig import Config
from classifier.utils.process import remove_punctuation, convert_to_lowercase

config = Config().config    # 获取配置文件
logging.basicConfig(level=logging.INFO)     # 配置日志文件输出级别

def load_dataset(file_path):
    """
    读取数据集中的内容
    :param file_path: 文件路径
    :return: sentence句子， labels标签
    二者都为List列表，且内容一一对应。
    """
    mode = file_path.split('/')[-1].split(".")[0]     # 读取类别为: test, train, valid三种
    sentence = []
    labels = []
    if mode in ['train', 'valid']:
        with open(Config().label2dict_path, "r", encoding="utf-8") as f:
            labels_mapping = yaml.load(f, Loader=yaml.FullLoader)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"正在加载 {mode} 数据集: ", total=len(lines)):
                label2num = []
                line = json.loads(line)
                word_list = convert_to_lowercase(remove_punctuation(line['text'])).split(' ')
                sentence.append(word_list)
                for label in line['label']:     # 将label映射为0-30对应的类别，详情请见/datasets/labelList.yaml文件
                    label2num.append(labels_mapping[label])
                labels.append(label2num)
            return sentence, labels
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"正在加载 {mode} 数据集: ", total=len(lines)):
                word_list = convert_to_lowercase(remove_punctuation(line)).split(' ')
                sentence.append(word_list)
            return sentence, None


def build_curpus(text, mode):
    """
    根据训练集文本构建词典
    :param text: 训练集
    :return: 词典, Embedding
    """
    if mode not in ['train']:   # 如果mode不为train, 则退出该部分
        return None, None
    word2index = {"<PAD>": 0, "<UNK>": 1}   # PAD代表填充字符     UNK代表一些没有见过的字符
    for t in tqdm(text, desc="正在构建词典: ", total=len(text)):
        for word in t:
            word2index[word] = word2index.get(word, len(word2index))
    words_dict_path = Config().words_dict_path
    with open(words_dict_path, 'w') as f:
        json.dump(word2index, f)
    logging.log(logging.INFO, f"词典构建完毕, 共计: {len(word2index)} 词, 已存储至 {words_dict_path}。 ")
    return word2index, nn.Embedding(len(word2index), config['embedding_dim'])


class TextDataset(Dataset):
    """
    读取数据集
        mode: 有三种选项
            1. train 读取训练集
            2. val 读取验证集
            3. test 读取测试集
    """

    def __init__(self, dataset, labels, word2index, mode='train'):
        """
        初始化函数
        :param word2index:
        :param mode:
        :param size:
        """
        assert mode in ['train', 'val', 'test'], 'mode should be train or val or test'
        super(TextDataset, self).__init__()
        self.max_length = config['max_length']      # 单句子最大长度
        self.word2index = word2index
        self.dataset, self.labels = dataset, labels
        self.mode = mode

    def __getitem__(self, idx):
        text = self.dataset[idx][:self.max_length]    # 如果句子长度大于max_length则对其进行截取
        text_idx = [self.word2index.get(i, 1) for i in text]     # 将单词转化为编码, 使用get方法是因为如果词在词典中不存在的话，用1进行替换，1定义为<UNK>
        text_idx = text_idx + [0] * (self.max_length - len(text_idx))   # 如果句子长度小于max_length, 补全到max_length长度
        if self.mode in ['train', 'val']:
            # 由于时多标签任务，所以采用独热码的方式每个label共计31位, 确保labels长度一样
            label = self.labels[idx]
            one_hot_label = torch.zeros(config['nc'])
            one_hot_label[label] = 1
            return torch.tensor(text_idx).unsqueeze(dim=0), one_hot_label
        else:
            return torch.tensor(text_idx).unsqueeze(dim=0)

    def __len__(self):
        return len(self.dataset)
