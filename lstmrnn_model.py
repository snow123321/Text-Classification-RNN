# -*- coding: utf-8 -*-


'''-------------------------------------------RNN模型------------------------------------------------'''

import tensorflow as tf


'''RNN配置参数'''
class RNNConfig(object):
    #模型参数
    embendding_dim = 100   #词向量维度
    seq_length = 600     #序列长度
    num_classes = 10     #类别数
    vocab_size = 5000    #词汇表大小
    
    n_layers = 2       #隐藏层层数
    n_neurons = 128     #隐藏层神经元数量
    rnn = "lstm"          #lstm或gru
    
    dropout_keep_prob = 0.5   #dropout保留比例
    learning_rate = 1e-3      #学习率
    
    batch_size = 128     #每批训练大小
    n_epochs = 5     #总迭代轮次
    
    print_per_batch = 100     #每多少轮输出一次结果
    save_per_batch = 1000       #每多少轮存入tensorboard
    
    
class TextRNN(object):
    def __init__(self,config):
        self.config = config
        
        self.input_x = tf.placeholder(tf.int32,[None,config.seq_length])
        self.input_y = tf.placeholder(tf.float32,[None,config.num_classes])
        self.dropout_prob = tf.placeholder(tf.float32,name = "dropout_prob")
        
        self.rnn()
        
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.config.n_neurons,state_is_tuple=True)
        
        def gru_cell():
            return tf.contrib.rnn.GRUCell(num_units=self.config.n_neurons)
        
        def basic_cell():
            return tf.contrib.rnn.BasicRNNCell(num_units=self.config.n_neurons)
        
        def dropout_cell():
            if self.config.rnn == "lstm":
                cell = lstm_cell()
            else:
                if self.config.rnn == "gru":
                    cell = gru_cell()
                else:
                    cell = basic_cell()
                
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_prob)
        
        #词向量映射
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",shape=[self.config.vocab_size,self.config.embendding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding,self.input_x)
            
        with tf.name_scope("rnn"):
            #多层rnn网络
            cells = [dropout_cell() for layer in range(self.config.n_layers)]
            rnn_cells = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(rnn_cells,inputs = embedding_inputs,dtype = tf.float32)
            last_outputs = outputs[:,-1,:]   #取最后一个时序输出结果
            
            #全连接层，后接dropout以及relu激活
            fc = tf.layers.dense(last_outputs,self.config.n_neurons,name = "fc1")
            fc = tf.contrib.layers.dropout(fc,self.dropout_prob)
            fc = tf.nn.relu(fc,name = "relu")
            
            #输出层
            self.logits = tf.layers.dense(fc,self.config.num_classes,name = "softmax")
            self.y_pred = tf.argmax(self.logits,1)
            
        #训练
        with tf.name_scope("training_op"):
            #损失函数，交叉熵
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(xentropy)
            
            #优化
            optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)
            
        #计算准确率
        with tf.name_scope("accuracy"):
            correct = tf.equal(self.y_pred,tf.argmax(self.input_y,1))
            self.acc = tf.reduce_mean(tf.cast(correct,tf.float32))
        
        
        
        
        
        
        
        
    
        
