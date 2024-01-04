"""
获取dataLoader
"""
import json

from torch.utils.data import DataLoader

from classifier.conf.readConfig import Config
from classifier.data.dataset import load_dataset, build_curpus, TextDataset

config = Config().config

def getDataLoader(batch_size, shuffle=False, mode='train'):
    """
    获取dataLoader
    :param batch_size:  批次大小
    :param shuffle:  是否随机打乱
    :param mode: 模式
    :return: DataLoader & embedding_matrix
    """
    assert mode in ['train', 'val', 'test'], f"mode should be 'train' , 'val' or 'test'"
    global data_path, word2index
    if mode == 'test':
        data_path = Config().test_path
    else:
        data_path = Config().train_path if mode == 'train' else Config().valid_path  # 读取数据集路径地址
    sentence, labels = load_dataset(data_path)  # 读取数据集
    if mode == 'train':     # 如果是训练集则重新构建字典
        word2index, embedding_matrix = build_curpus(sentence, mode)  # 获取数据集的字典
    else:   # 如果是测试集或者是验证集 则读取字典
        with open(Config().words_dict_path, 'r') as file:
            word2index = json.load(file)
    dataset = TextDataset(dataset=sentence, labels=labels, word2index=word2index, mode=mode)
    if mode == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config['num_workers']
        ), embedding_matrix
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
