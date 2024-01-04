import logging
import os
import threading

import torch
from torch import nn
from tqdm import tqdm

from classifier.conf.readConfig import Config
from classifier.data.loaders import getDataLoader
from classifier.engine.validator import validator, last_validator, predict_test
from classifier.nn.model import TextCNNModel
from classifier.utils.plotting import plot_loss_acc_curve
from classifier.utils.save import save_model, save_log

save_path = Config().save_path

def trainer(model_path=None):
    global loss, thread_save_best, net
    best_accuracy = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    item_list = os.listdir(save_path)
    s_p = save_path + '/run_' + str(len(item_list))    # 这是最终的保存路径
    logging.info(f"训练结果保存的路径为: " + s_p)

    loss_list = []
    accuracy_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config().config
    dataloader, embedding_matrix = getDataLoader(batch_size=config['batch_size'], mode='train', shuffle=True)

    val_dataloader = getDataLoader(batch_size=config['batch_size'], mode='val', shuffle=False)

    if model_path is None:
        net = TextCNNModel(embedding_matrix=embedding_matrix).to(device)
    else:
        net = torch.load(model_path).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    logging.info(f"开始训练... 训练总轮数为: {config['epoch']}")
    for epoch in range(config['epoch']):
        logging.info(f"================================epoch:{epoch+1}/{config['epoch']}================================")
        for x, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epoch']}"):
            x, label = x.to(device), label.to(device)
            pred = net(x)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info(f"Loss: {loss.item():.8f}")

        acc = validator(net, val_dataloader, device)
        logging.info(f"Accuracy: {acc:.8f}")

        # 保存最优模型 以多线程的方式保存，加快程序运行速度
        if acc > best_accuracy:
            best_accuracy = acc
            thread_save_best = threading.Thread(target=save_model, args=(net, "best.pt", s_p))
            thread_save_best.start()
        # 保存最后一次训练的模型
        thread = threading.Thread(target=save_model, args=(net, "last.pt", s_p))
        thread.start()
        loss_list.append(loss.item())
        accuracy_list.append(acc)
        save_log(s_p, f"epoch: {epoch+1} : loss: {loss.item():.8f} \t accuracy: {acc:.8f}")

    logging.info(f"等待模型存储线程保存模型完毕...")
    thread_save_best.join()
    # 验证一下最好的模型
    best_net_path = os.path.join(s_p, "weights", "best.pt")
    best_net = torch.load(best_net_path)  # 读取最好的模型
    logging.info(f"正在验证最好的模型, 模型路径为: " + best_net_path )
    best_acc = last_validator(best_net, val_dataloader, device, s_p)
    logging.info(f"✨验证集上最优准确率为: {best_acc:.8f}")
    save_log(s_p, f"✨验证集上的最优准确率为: {best_acc:.8f}")
    save_log(s_p, f"🎈最优模型权重已保存至: {best_net_path}")

    # 对测试集(test)进行预测并进行存储预测值
    test_dataloader = getDataLoader(config['batch_size'], shuffle=False, mode='test')
    predict_test(best_net, test_dataloader, device, s_p)
    logging.info(f"🎈对测试集预测完毕, 文件保存在: {s_p}")
    save_log(s_p, f"✌️测试集的预测结果已存储在: {s_p}/test_predict(submission).txt")


    # 执行结束后绘制损失函数曲线以及准确率曲线
    plot_loss_acc_curve(loss_list, accuracy_list, s_p)

