"""
用于绘制损失函数曲线，以及准确率曲线
"""
import os

import matplotlib.pyplot as plt

def plot_loss_acc_curve(loss, accuracy, save_path):
    """
    绘制损失函数曲线，和准确率曲线
    :param loss: 损失值 type: list
    :param accuracy: 准确率值 type: list
    :return:
    """
    epochs = list(range(len(loss)+1))[1:]
    # 创建一个图形对象
    fig, ax1 = plt.subplots()
    # 绘制损失函数曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, loss, color='tab:blue', label='Loss Curve')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # 创建第二个坐标轴对象，共享x轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.plot(epochs, accuracy, color='tab:red', label='Accuracy Curve')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # 添加标题
    plt.title('Loss and Accuracy Curves')
    # 添加图例
    fig.tight_layout()
    fig.legend(loc='upper left')
    # 保存图形
    plt.savefig(os.path.join(save_path, 'loss_accuracy_curves.png'))
    # 显示图形
    plt.show()
