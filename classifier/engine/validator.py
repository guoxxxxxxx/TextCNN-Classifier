"""
用于验证验证集 val
"""
import torch
from tqdm import tqdm

from classifier.conf.readConfig import Config
from classifier.data.loaders import getDataLoader
from classifier.utils.save import save_result

config = Config().config


def validator(net, val_dataloader, device):
    """
    用于每个epoch结束后对预测集进行评估
    :param net: 网络模型
    :param val_dataloader: 验证集的dataloader
    :param device: 设备
    :return:
    """
    with torch.no_grad():
        acc_list = []
        for x, label in tqdm(val_dataloader, desc='Validating: '):
            x, label = x.to(device), label.to(device)
            pred = net(x)
            # 通过sigmoid映射之后值就分布在了0，1之间，相当于概率，当概率小于0.5时，则认为其不属于这个类别。大于0.5时则认为其属于这个类别
            bin_pred = (torch.sigmoid(pred) > 0.5).float()
            # 判断每句话的每个类别是否预测正确
            correct = bin_pred == label
            # 下面这段的意思时，只有这句话的类别都预测正确，才算正确。
            correct = correct.all(dim=1).float()
            ave_acc = correct.sum().item() / len(label)
            acc_list.append(ave_acc)
        # 返回准确率
        return sum(acc_list) / len(acc_list)


def last_validator(net, val_dataloader, device, save_path):
    """
    和上述方法一样，就是这个方法将最终结果保存到txt了，用于最后的模型评估
    :param net: 网络模型
    :param val_dataloader: 验证集的dataloader
    :param device: 设备
    :param save_path: 保存的路径
    :return:
    """
    val_res_list = []
    with torch.no_grad():
        acc_list = []
        for x, label in tqdm(val_dataloader, desc='Validating: '):
            x, label = x.to(device), label.to(device)
            pred = net(x)
            bin_pred = (torch.sigmoid(pred) > 0.5).float()
            # 这部分内容主要是用于保存模型的最后预测结果
            part_res_list = [torch.nonzero(row).flatten().tolist() for row in bin_pred]
            val_res_list += part_res_list
            correct = bin_pred == label
            correct = correct.all(dim=1).float()
            ave_acc = correct.sum().item() / len(label)
            acc_list.append(ave_acc)
        # 保存到txt文件 这个文件是验证集的预测，不是最终提交的文件
        save_result(val_res_list, filename="val_predict(no submission).txt", save_path=save_path, log="验证集(validation)")
        return sum(acc_list) / len(acc_list)


def predict_test(net, test_loader, device, save_path):
    """
    对测试集进行预测
    :param net: 模型(此处传入的是最优模型)
    :param test_loader: test dataloader
    :param device: 设备
    :param save_path: 保存路径
    :return:
    """
    test_res_list = []
    with torch.no_grad():
        for x in tqdm(test_loader, desc='Validating: '):
            x = x.to(device)
            pred = net(x)
            bin_pred = (torch.sigmoid(pred) > 0.5).float()
            # 这部分内容主要是用于保存模型的最后预测结果
            part_res_list = [torch.nonzero(row).flatten().tolist() for row in bin_pred]
            test_res_list += part_res_list
        # 保存到txt文件 这部分内容是最终提交的
        save_result(test_res_list, filename="test_predict(submission).txt", save_path=save_path, log="测试集(test)")

