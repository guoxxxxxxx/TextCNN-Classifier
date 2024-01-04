"""
用于保存模型 和 测试集及验证集结果保存
"""
import os

import torch
import yaml
from tqdm import tqdm

from classifier.conf.readConfig import Config



def save_model(net, filename, save_path):
    """
    保存模型
    :param net: 模型
    :param filename: 保存的名称
    :param save_path: 保存的路径
    :return:
    """
    if not os.path.exists(save_path + "/weights/"):
        os.makedirs(save_path + "/weights/", exist_ok=True)
    torch.save(net, save_path + "/weights/" + filename)


def save_result(res_list, filename, save_path, log):
    """
    将预测结果转为标签值存入txt
    :param res_list: 结果列表
    :param filename: 文件名
    :param save_path: 保存路径
    :param log: 日志
    :return:
    """
    global dict2label_mapping

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_save_path = os.path.join(save_path, filename)

    # 读取映射表
    with open(Config().dict2label_path, "r", encoding="utf-8") as f:
        dict2label_mapping = yaml.load(f, Loader=yaml.FullLoader)

    with open(file_save_path, "w") as f:
        for res in tqdm(res_list, desc=f"正在保存 {log} 中的内容", total=len(res_list)):
            # 将列表中的每个元素转换为字符串，以逗号分隔，并在末尾添加换行符
            mapped_values = [dict2label_mapping.get(key, None) for key in res]
            line = ','.join(map(str, mapped_values))
            f.write(line + '\n')


def save_log(save_path, log):
    """
    保存日志
    :param save_path: 保存路径
    :param log: 保存内容
    :return:
    """
    with open(os.path.join(save_path, 'log.txt'), 'a', encoding="utf-8") as f:
        f.write(log + '\n')

