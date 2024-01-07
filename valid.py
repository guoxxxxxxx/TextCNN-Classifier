# 用于使用指定模型验证验证集
import logging
import os

import torch

from classifier.conf.readConfig import Config
from classifier.data.loaders import getDataLoader
from classifier.engine.validator import last_validator
from classifier.utils.save import save_log

config = Config().config
save_val_path = Config().save_val_path

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 要使用的模型，例如: runs/train/run_0/weights/best.pt
    model_path = "runs/train/run_0/weights/best.pt"

    if not os.path.exists(save_val_path):
        os.makedirs(save_val_path)
    item_list = os.listdir(save_val_path)
    s_p = save_val_path + '/val_' + str(len(item_list))    # 这是最终的保存路径
    logging.info(f"训练结果保存的路径为: " + s_p)

    val_dataloader = getDataLoader(batch_size=config['batch_size'], mode='val', shuffle=False)
    best_net = torch.load(model_path)  # 读取最好的模型
    logging.info(f"正在验证模型, 模型路径为: " + model_path)
    best_acc = last_validator(best_net, val_dataloader, device, s_p)
    logging.info(f"✨验证集上的确率为: {best_acc:.8f}")
    save_log(s_p, f"所使用的模型路径为: {model_path}")
    save_log(s_p, f"✨验证集上的准确率为: {best_acc:.8f}")
