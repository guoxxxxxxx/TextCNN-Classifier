"""
读取配置文件
"""
import os

import yaml


absPath = os.path.abspath(".")
# 配置文件路径
config_path = absPath + "/config/config.yaml"

class Config:
    """
    配置类, 读取配置文件，并将路径信息从相对路径转化为绝对路径。
    """

    def __init__(self):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.absPath = absPath

    @property
    def train_path(self):
        return self.absPath + self.config["train_path"]

    @property
    def test_path(self):
        return self.absPath + self.config["test_path"]

    @property
    def valid_path(self):
        return self.absPath + self.config["valid_path"]

    @property
    def label2dict_path(self):
        return self.absPath + self.config["label2dict_path"]

    @property
    def save_predict_path(self):
        return self.absPath + self.config["save_predict_path"]

    @property
    def save_val_path(self):
        return self.absPath + self.config["save_val_path"]

    @property
    def dict2label_path(self):
        return self.absPath + self.config["dict2label_path"]

    @property
    def words_dict_path(self):
        return self.absPath + self.config["words_dict_path"]

    @property
    def word2index_path(self):
        return self.absPath + self.config["word2index_path"]

    @property
    def save_path(self):
        return self.absPath + self.config["save_path"]

    @property
    def max_length(self):
        return self.config["max_length"]

    @property
    def hyper_params(self):
        return self.config

