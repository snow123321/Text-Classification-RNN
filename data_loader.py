# -*- coding: utf-8 -*-


'''--------------------------------------文本数据处理------------------------------------------------------'''

from collections import Counter
import numpy as np
import tensorflow.keras as kr

#读取文件数据
def read_file(file_path):
    f = open(file_path,"r",encoding = "utf-8")
    data = f.readlines()
    f.close()
    
    labels = []
    contents = []
    for line in data:
        label,content = line.strip().split("\t")
        labels.append(label)
        contents.append(list(content))
    
    return labels,contents
  

#将词汇表中的词转换为id {词：id}，dictionary形式
def bulid_vocab_id():
    '''原本参考的例子中词汇表第一个词为手动添加的“<PAD>”，这样转换时word_id为0表示<PAD>这个词，
    在实际语料中不存在，因为后面会把所有句子都padding为同样的长度，padding填充的是0，这样在后面训练
    词向量的时候，所有句子padding的部分的词向量都是一样的。
    也可在设置输入序列长度可变，在训练时输入每个句子的实际长度'''
    
    f = open("./cnews_data/cnews.vocab.txt","r",encoding = "utf-8")
    vocab = f.readlines()
    f.close()
    vocab = list(set([word.strip() for word in vocab]))
    
    vocab_id = dict(zip(vocab,range(len(vocab))))
    
    return vocab,vocab_id


#将类别转换为id {类别：id},dictionary形式
def build_category_id():
    categories = ["体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"]
    #将类别赋予编号（id）
    cat_to_id = dict(zip(categories,range(len(categories))))
    
    return cat_to_id


#将文本转换为id表示
def word_to_id(file_path,vocab_size,max_length):
    vocab,vocab_id = bulid_vocab_id()
    cat_to_id = build_category_id()
    
    labels,contents = read_file(file_path)
    words_to_id = []
    labels_to_id = []
    for i in range(len(contents)):
        words_to_id.append([vocab_id[word] for word in contents[i] if word in vocab_id])
        labels_to_id.append(cat_to_id[labels[i]])
        
    #使用keras提供的pad_sequence来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(words_to_id,max_length)
    y_pad = kr.utils.to_categorical(labels_to_id,num_classes=len(cat_to_id))   #将y转换成one-hot向量
    
    return x_pad,y_pad


#生成批次数据
def batch_iter(x,y,batch_size):
    num_batch = int((len(x)-1) / batch_size) + 1
    
    indices = np.random.permutation(np.arange(len(x)))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size,len(x))
        
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]
        
        
    
    
        
    
    



