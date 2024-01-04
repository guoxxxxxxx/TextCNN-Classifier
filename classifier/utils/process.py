"""
数据集预处理文件，具体包含方法如下：
    1. remove_punctuation(text): 去除文本中的标点符号
    2. convert_to_lowercase(text): 将文本中的大写字母全部转化为小写字母
"""
import json
import string

import matplotlib.pyplot as plt
from tqdm import tqdm

from classifier.conf.readConfig import Config


def remove_punctuation(text):
    """
    去除文本中的标点符号
    :param text: 所要处理的文本
    :return: 去除标点符号的文本
    """
    translator = str.maketrans('', '', string.punctuation)
    text_without_punc = text.translate(translator)
    return text_without_punc


def convert_to_lowercase(text):
    """
    将文本中的大写字母全部转化为小写字母
    :param text: 所以转换的文本
    :return: 转换后的文本
    """
    text_lowercase = text.lower()
    return text_lowercase


def text_len_list():
    """
    绘制每个句子单词的分布图
    :return:
    """
    from classifier.data import dataset
    text_len = []
    sentence, _ = dataset.load_dataset(Config().train_path)
    count = 0
    exceed_sentence = []
    threshold_value = 500
    for i, line in enumerate(sentence):
        text_len.append(len(line.split()))
        if text_len[i] > threshold_value:
            exceed_sentence.append(i)
            count += 1
            text_len[i] = 0
    plt.hist(text_len, bins=200)
    plt.show()
    print("超过阈值", str(threshold_value), "的句子共有", str(count), "个")
    print("其下标为: ", exceed_sentence)


def save_label2txt():
    with open("./valid_label.txt", 'w', encoding='utf-8') as w:
        with open(Config().valid_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"正在加载 valid 数据集: ", total=len(lines)):
                line = json.loads(line)
                w.write(','.join(line['label']) + '\n')




