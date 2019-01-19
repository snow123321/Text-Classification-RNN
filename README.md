# 学习笔记：用RNN进行中文文本分类
本文是基于TensorFlow使用RNN进行的中文文本分类<br>
## 环境
Python 3<br>
TensorFlow<br>
## 数据集
使用THUCNews的一个子集进行训练与测试（由于数据集太大，无法上传到Github，可自行下载），数据集划分如下：<br>
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
在3epoch时停止训练，最终在测试集上准确率85%。可以通过调参以达到更高的准确率<br><br>
<img src="https://github.com/snow123321/Text-Classification-RNN/blob/master/images/image1_resutle.jpg" width="400" height="350"><br>
## 学习过程中遇到的问题
（由于是刚开始学，也是参考的别人的例子然后自己基本是重复写了一遍，这个过程中刚开始有很多不明白的地方，通过查资料最后才弄明白）<br>
### 1、padding及embedding
词汇表中第一个词为手动添加的“\<PAD>”，这样转换时word_id为0表示<PAD>，这个词在实际语料中是不存在的。是因为后面会把所有句子都padding为同样的长度，padding填充的是0，这样在后面训练词向量的时候，所有句子padding的部分的词向量都是一样的。<br>
 此次试验用的是字符级词向量（char embedding），调用tensorflow的API实现。<br>
```Python
embedding = tf.get_variable("embedding",shape=[self.config.vocab_size,self.config.embendding_dim])
embedding_inputs = tf.nn.embedding_lookup(embedding,self.input_x)
```
 
 **embedding是在最开始随机初始化，然后在模型训练过程中一起训练的！**<br>
 tf.nn.embedding_lookup的作用是在embedding中根据输入的词id查找对应的向量。这里也可以使用预训练好的词向量，在训练时就不再训练词向量了，这样还可以设置输入序列长度可变，后面准备试试这种方法。<br>

### 2、outputs和 states
 ```Python
outputs, states = tf.nn.dynamic_rnn(multi_cells, X, dtype = tf.float32)
last_outputs = outputs[:,-1,:]   #取最后一个时序输出结果
```

outputs大小为[batch_size,n_steps,n_inputs],即和输入X大小一样；<br>
states为最终状态，一般情况下大小为[batch_size,cell.output_size],但当cell为lstm时，大小为[2,batch_size,cell.output_size]，分别对应lstm的cell state和hidden state。<br>
**当有多个隐藏层时，outputs是最后一层的输出，states是每一层最终状态的输出。**

 
 




