# NLP



## 各种模型对比

### ① TextCNN

- 跑了50轮实验结果如下

  - 损失函数及精度曲线如下(验证集上的精度在85%左右)

    <img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281754131.png" alt="image-20231228172405338" style="zoom: 33%;" />

  - 训练日志文件如下：

    ``` txt
    ... 上述轮数省略, 请见项目文件
    epoch: 48 : loss: 0.00112865 	 accuracy: 0.85228954
    epoch: 49 : loss: 0.00047265 	 accuracy: 0.83601856
    epoch: 50 : loss: 0.00248988 	 accuracy: 0.85001596
    ✨验证集上的最优准确率为: 0.85326610
    🎈最优模型权重已保存至: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0\weights\best.pt
    ✌️测试集的预测结果已存储在: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0/test_predict(submission).txt
    ```

  - 超参数设定如下：

    ``` yaml
    # 超参数设定
    max_length: 500   # 句子的最长长度
    embedding_dim: 256  # 每个单词用几个值表示
    batch_size: 512    # 训练时每个批次的大小
    epoch: 50     # 训练的轮数
    learning_rate: 0.001   # 学习率
    num_workers: 0    # 读取数据的进程数
    
    # 模型相关参数设定
    nc: 31    # 类别数
    hidden_layer: 256   # 隐藏层数量(通道数)
    ```

    

### ② LSTM

跑了50轮实验结果如下

- 损失函数及精度曲线如下(验证集上的精度在85%左右)

  <img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281754131.png" alt="image-20231228172405338" style="zoom: 33%;" />

- 训练日志文件如下：

  ``` txt
  ... 上述轮数省略, 请见项目文件
  epoch: 48 : loss: 0.00112865 	 accuracy: 0.85228954
  epoch: 49 : loss: 0.00047265 	 accuracy: 0.83601856
  epoch: 50 : loss: 0.00248988 	 accuracy: 0.85001596
  ✨验证集上的最优准确率为: 0.85326610
  🎈最优模型权重已保存至: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0\weights\best.pt
  ✌️测试集的预测结果已存储在: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0/test_predict(submission).txt
  ```

- 超参数设定如下：

  ``` yaml
  # 超参数设定
  max_length: 500   # 句子的最长长度
  embedding_dim: 256  # 每个单词用几个值表示
  batch_size: 512    # 训练时每个批次的大小
  epoch: 50     # 训练的轮数
  learning_rate: 0.001   # 学习率
  num_workers: 0    # 读取数据的进程数
  
  # 模型相关参数设定
  nc: 31    # 类别数
  hidden_layer: 256   # 隐藏层数量(通道数)
  ```

  

### ③ LSTM+TextCNN



> 先进行LSTM再进行TextCNN



跑了50轮实验结果如下

- 损失函数及精度曲线如下(验证集上的精度在85%左右)

<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202401071743188.png" alt="image-20240107174331131" style="zoom: 33%;" />

- 训练日志文件如下：

  ``` txt
  epoch: 47 : loss: 0.00045492 	 accuracy: 0.84806283
  epoch: 48 : loss: 0.00092862 	 accuracy: 0.84253658
  epoch: 49 : loss: 0.00103872 	 accuracy: 0.85164610
  epoch: 50 : loss: 0.00079300 	 accuracy: 0.85034401
  ✨验证集上的最优准确率为: 0.85164610
  🎈最优模型权重已保存至: F:\Programs\Remote_Project\TextCNN\TextCNN-Classifier/runs/train/run_5\weights\best.pt
  ✌️测试集的预测结果已存储在: F:\Programs\Remote_Project\TextCNN\TextCNN-Classifier/runs/train/run_5/test_predict(submission).txt
  ```

- 超参数设定如下：

  ``` yaml
  # 超参数设定
  max_length: 500   # 句子的最长长度
  embedding_dim: 256  # 每个单词用几个值表示
  batch_size: 512    # 训练时每个批次的大小
  epoch: 50     # 训练的轮数
  learning_rate: 0.001   # 学习率
  num_workers: 0    # 读取数据的进程数
  
  # 模型相关参数设定
  nc: 31    # 类别数
  hidden_layer: 256   # 隐藏层数量(通道数)
  
  # LSTM模块参数
  lstm_hidden_layer: 256
  ```

  





## 0. 前言



本项目为华北电力大学2023学年度自然语言处理结课作业



> 主要参考项目视频讲解：[🔗Bilibili](https://www.bilibili.com/video/BV163411w7qg?p=1&vd_source=92b2eaa496fe004107c04c4f55ee0fca)
>
> 项目开源地址：[🔗GitHub](https://github.com/shouxieai/A-series-of-NLP/blob/main/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/TextCNN_%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/textCNN_%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.py)





## 1. 详细实现步骤

> 说明：baseline就不阐述了，可以看前言的资料。 这部分主要说针对这个作业需要对上述模型进行修改的部分。



- 网络结构与这个大体相同，在max-pooling的基础上，~~做了个分支，添加了ave-pooling。~~(效果有提升，但是不明显，计算开销明显增大，故删除此部分)



<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202401071745515.png" alt="image-20240107174538423" style="zoom: 33%;" />

### 1.1 数据集读取部分



- 首先观察数据集，有些文本对应的标签并不唯一，有一个文本对应多个标签的情况。所以该问题是一个多标签分类问题。

<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281557467.png" alt="image-20231226082033933" style="zoom: 50%;" />



- 由于该问题时多标签分类问题，所以Dataset中返回的label比较适合采用one-hot编码来写，如果直接返回label则回报错，因为其是边长的，采用one-hot编码可以有效解决该问题。代码详见: $classifier/data/dataset.py$

  

- 通过分布可以看出训练集中绝大多数句子都在500词以下（通过验证只有6个句子在500词以上）所以每个句子的最大长度取500。代码详见: $classifier/data/dataset.py$

  

<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281557391.png" alt="image-20231226193424361" style="zoom: 80%;" />

<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281557432.png" alt="image-20231226194442358" style="zoom:50%;" />



``` python
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
```




### 1.2 输出头部分修改



- 由于共有31个类别，且为多标签分类，所以在最后的输出上应该有31个输出，表示每个类别的可能性。这部分主要在$classifier/nn/model.py$文件中体现。其中$config['nc']$代表类别数。

  <img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281557472.png" alt="image-20231228155707037" style="zoom:50%;" />

  - 每行文本经模型预测得到的输出有31个值，分别代表31个类别的概率。这31个值的目前是在整个实数域上，也就是取值为负无穷到正无穷。所以损失函数使用$BCEWithLogitsLoss$作为损失函数，相当于在做损失函数之前，做了一下$Sigmoid$。代码实现请见文件夹$classifier/engine/trainer.py$



- 对于验证集验证部分，通过训练的模型对验证集进行验证时，输出和上述情况一样，所以也需要进行一下$Sigmoid$操作，将预测值映射到$[0, 1]$之间。代码请见文件$classifier/engine/validator.py$

  ``` python
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
  ```




### 1.3 超参数修改

- 如果修改超参数请见文件$config/config.yaml$

  ``` yaml
  # 配置文件
  label2dict_path: /datasets/label2dict.yaml  # 标签到字典的映射
  dict2label_path: /datasets/dict2label.yaml  # 字典映射到标签文件
  words_dict_path: /datasets/words_dict.json  # 字典保存路径
  train_path: /datasets/train.json  # 数据集的路径
  valid_path: /datasets/valid.json  # 验证集的路径
  test_path: /datasets/test.txt   # 测试集的路径
  word2index_path: /datasets/word2index.yaml  # word2index文件 目前没用
  save_path: /runs/train    # 模型保存路径
  save_predict_path: /runs/predict  # 预测时保存的路径
  save_val_path: /runs/valid    # 单独运行验证程序时保存的路径
  
  # 超参数设定
  max_length: 500   # 句子的最长长度
  embedding_dim: 256  # 每个单词用几个值表示
  batch_size: 512    # 训练时每个批次的大小
  epoch: 1     # 训练的轮数
  learning_rate: 0.001   # 学习率
  num_workers: 0    # 读取数据的进程数
  
  # 模型相关参数设定
  nc: 31    # 类别数
  hidden_layer: 256   # 隐藏层数量(通道数)
  ```

  



## 2. 程序运行详情



- 步骤1： 解压文件，并进入到文件夹

  <img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281709376.png" alt="image-20231228170921284" style="zoom: 33%;" />



- 步骤2：在命令行下顺序输入如下指令

``` cmd
conda create -n NLP python=3.11		# 创建新的环境
conda activate NLP		# 切换到新环境

pip install -r requirement.txt		# 安装相关依赖
```

> 提示： 如果安装依赖时，到安装torch包报错的话，可以把有关torch的那几行注释掉(前面加#号就行)，去官网安装
>
> 官网地址为: https://pytorch.org/get-started/locally/
>
> 选择对应的版本安装即可，

<img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281754732.png" alt="image-20231228171646955" style="zoom:33%;" />

- 步骤3：在Pycharm打开项目文件，选择刚才装好的环境即可运行



- 运行文件
  - 如果训练的话运行 $train.py$即可
  - 如果验证验证集的话运行$valid.py$即可
  - 如果预测测试集的内容话运行$predict.py$即可



## 3. 程序运行情况

> 这里以TextCNN为例，其他都一样



- 跑了50轮实验结果如下

  - 损失函数及精度曲线如下(验证集上的精度在85%左右)

    <img src="https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281754131.png" alt="image-20231228172405338" style="zoom: 33%;" />

  - 训练日志文件如下：

    ``` txt
    ... 上述轮数省略, 请见项目文件
    epoch: 48 : loss: 0.00112865 	 accuracy: 0.85228954
    epoch: 49 : loss: 0.00047265 	 accuracy: 0.83601856
    epoch: 50 : loss: 0.00248988 	 accuracy: 0.85001596
    ✨验证集上的最优准确率为: 0.85326610
    🎈最优模型权重已保存至: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0\weights\best.pt
    ✌️测试集的预测结果已存储在: F:\Programs\Python\AI\PyCharm\NLP/runs/train/run_0/test_predict(submission).txt
    ```



- 说明

  - $test\_predict(submission).txt$文件为预测验证集的文件，验证集的真实值可以看文件$datasets/valid\_label.txt$可以直观的做对比。部分对比如下图所示。

    ![image-20231228173124801](https://cdn.jsdelivr.net/gh/guoxxxxxxx/Pic-Go@main/img/202312281753893.png)

    - 说明：（左侧部分为真实值，右侧部分为预测值）

      红色框选部分：两边对不上，说明预测错误。

      蓝色部分：之所以预测值为空，是因为这是一个多标签分类，程序认为编号为47的语句不属于任何一类，所以类别为空值。



## 附录

项目文件结构

$NLP$ 项目根目录

- $classifier$ 包，整个程序的核心代码在此处
  - $conf$：读取配置的包
    - $readConfig.py$	配置脚本，这部分内容时为了读取$config.yaml$配置文件
  - $data$: 读取数据的包
    - $dataset.py$		数据读取脚本，这部分主要是读取数据
    - $loaders.py$   这部分是获取$dataloader$迭代器，更好的获取数据
  - $engine$: 训练和验证的代码包
    - $trainer.py$ 这部分是主要包含训练的代码
    - $validator.py$ 这部分主要包含验证验证集和预测测试集的代码
  - $nn$: 模型代码包
    - $model.py$ 此文件中是构建模型的代码
  - $utils$： 工具包
    - $plotting.py$ 绘制图像的代码
    - $process.py$ 对数据集进行简单查看的代码，如长度啊什么的，没用了现在，不用看这个
    - $save.py$ 模型保存以及日志文件保存的代码
- $config$文件夹
  - $config.yaml$	配置类
- $datasets$文件夹
  - $dict2label.yaml$	将字典转化为标签的映射文件
  - $label2dict.yaml$ 将标签转化为字典的映射文件
  - $label\_list.txt$  类别
  - $test.txt$ 测试集
  - $train.json$ 训练集
  - $valid.json$ 验证集
  - $valid\_label.txt$  验证集的单独标签版，主要是方便对照着看，程序里没用
  - $words_dict.json$ 程序自动构建的词典
- $runs$文件夹
  - $predict$ 单独执行$predict.py$时的保存路径
  - $train$ 执行$train.py$文件时保存的路径
  - $valid$ 执行$valid.py$时保存的路径



**重要的事情说三遍！！！**

**说明：最后提交作业时，提交$test\_predict(submission).txt$文件，这个是测试集的预测文件！**

**说明：最后提交作业时，提交$test\_predict(submission).txt$文件，这个是测试集的预测文件！**

**说明：最后提交作业时，提交$test\_predict(submission).txt$文件，这个是测试集的预测文件！**

