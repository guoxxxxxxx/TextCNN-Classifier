from classifier.engine.trainer import trainer
import torch
import numpy as np
import random

if __name__ == '__main__':

    # 设置PyTorch的随机数种子 保证模型可以复现
    seed = 666666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能影响性能，但有助于确保结果的一致性
    # 设置NumPy的随机数种子
    np.random.seed(seed)
    # 设置Python标准库的随机数种子
    random.seed(seed)

    # 如果想继续使用之前训练的模型训练的话，将model_path值更改为模型路径就行，比如说/runs/run_01/weights/best.pt
    # 如果为None则就是从0开始训练
    model_path = None
    trainer(model_path=model_path)
