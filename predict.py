# 使用已经训练好的模型对测试集进行预测时，可以单独执行该文件

import logging
import os

import torch

from classifier.conf.readConfig import Config
from classifier.data.loaders import getDataLoader
from classifier.engine.validator import predict_test
from classifier.utils.save import save_log

save_predict_path = Config().save_predict_path
config = Config().config

if __name__ == '__main__':
    # 使用的模型的路径
    model_path = "runs/train/run_0/weights/best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_predict_path):
        os.makedirs(save_predict_path)
    item_list = os.listdir(save_predict_path)
    s_p = save_predict_path + '/pred_' + str(len(item_list))    # 这是最终的保存路径
    logging.info(f"预测结果保存的路径为: " + s_p)

    # 读取模型
    net = torch.load(model_path).to(device)

    # 这里的shuffle一定要为False不然顺序就对不上了
    test_dataloader = getDataLoader(config['batch_size'], shuffle=False, mode='test')
    predict_test(net, test_dataloader, device, s_p)
    logging.info(f"🎈对测试集预测完毕, 文件保存在: {s_p}")
    save_log(s_p, f"✌️测试集的预测结果已存储在: {s_p}/test_predict(submission).txt")

