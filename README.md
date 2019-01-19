# 学习笔记：用RNN进行中文文本分类
本文是基于TensorFlow使用RNN进行的中文文本分类<br>
## 环境
Python 3<br>
TensorFlow<br>
## 数据集
使用THUCNews的一个子集进行训练与测试，数据集划分如下：<br>
训练集cnews.train.txt   50000条<br>
验证集cnews.val.txt   5000条<br>
测试集cnews.test.txt   10000条<br>
共分为10个类别："体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"。<br>
cnews.vocab.txt为词汇表，字符级，大小为5000，根据频次选择的。<br>
## 文件说明
data_loader.py：文本数据处理<br>
lstmrnn_model.py：RNN模型，实验的基本配置参数<br>
run_RNN.py：主函数文件<br>

## 结果

 




