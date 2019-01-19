# -*- coding: utf-8 -*-


'''------------------------------------主函数------------------------------------------------------'''

from data_loader import word_to_id,batch_iter,build_category_id
from lstmrnn_model import RNNConfig,TextRNN

import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
from sklearn import metrics
import gc


#公共函数
def get_time_dif(start_time):
    '''计算时间'''
    
    end_time = time.time()
    time_dif = end_time - start_time
    
    return timedelta(seconds = int(round(time_dif)))


def feed_data(x_batch,y_batch,dropout_prob):
    feed_dict = {
            model.input_x:x_batch,
            model.input_y:y_batch,
            model.dropout_prob:dropout_prob
            }
    return feed_dict
    


#评估在验证集或测试集上的准确率和损失，输入的是全部验证集或测试集数据。
def evaluate(sess,x,y):
    data_len = len(x)
    batch_eval = batch_iter(x,y,128)
    
    total_acc = 0.0
    total_loss = 0.0
    
    for x_batch,y_batch in batch_eval:
        feed_dict = feed_data(x_batch,y_batch,1.0)
        #算出来的loss和acc是在这一批次数据上的均值
        loss,acc = sess.run([model.loss,model.acc],feed_dict = feed_dict)
        
        total_loss = total_loss + loss*len(x_batch)
        total_acc = total_acc + acc*len(x_batch)
    
    return total_loss/data_len,total_acc/data_len
    


def train(train_dir,val_dir):
    #最佳验证模型保存路径
    save_dir = "./best_model"
    save_path = os.path.join(save_dir,"best_validation")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
        
    #配置Tensorboard
    #收集训练过程中的loss和acc
    tensorboard_dir = "./tensorboard"
    tf.summary.scalar("loss",model.loss)
    tf.summary.scalar("accuracy",model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    
    #获得训练和验证数据
    print("Loading train and val data...................")
    start_time = time.time()
    x_train,y_train = word_to_id(train_dir,vocab_size=config.vocab_size,max_length=config.seq_length)
    x_val,y_val = word_to_id(val_dir,vocab_size=config.vocab_size,max_length=config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:",time_dif)


    print("Training and evaluating................................")
    start_time = time.time()
    total_batch = 0   #训练批次
    best_val_acc = 0.0   #最佳验证集准确率
    last_improved_batch = 0   #上一次验证集准确率提升时的训练批次
    require_improved = 800   #若超过800轮次没有提升，则提前结束训练
    flag = False   #标识是否需要提前结束训练

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch in range(config.n_epochs):
            print("Epoch:",epoch)
            batch_train = batch_iter(x_train,y_train,config.batch_size)
            for x_batch,y_batch in batch_train:
                feed_dict = feed_data(x_batch,y_batch,dropout_prob=config.dropout_keep_prob)
                
                #每多少轮次将数据写入tensorboard
                if total_batch % config.save_per_batch == 0:
                    summary = sess.run(merged_summary,feed_dict = feed_dict)  
                    writer.add_summary(summary,total_batch)
                
                
                #每多少轮输出在训练集和验证集上的损失及准确率
                if total_batch % config.print_per_batch == 0:
                    #训练集损失和准确率
                    feed_dict[model.dropout_prob] = 1.0
                    train_loss,train_acc = sess.run([model.loss,model.acc],feed_dict = feed_dict)
                    #验证集损失和准确率
                    val_loss,val_acc = evaluate(sess,x_val,y_val)
                    #保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        saver.save(sess,save_path)
                        last_improved_batch = total_batch
                        improved_str = "***"
                    else:
                        improved_str = ""
                        
                    time_dif = get_time_dif(start_time)
                    msg = "Iter:{0:>4},Train loss:{1:>6.2}, Train accuracy:{2:>6.2%}, Val loss:{3:>6.2}, Val accuracy:{4:>6.2%}, Time usage:{5}  {6}"
                    print(msg.format(total_batch,train_loss,train_acc,val_loss,val_acc,time_dif,improved_str))
                
                #优化
                sess.run(model.optim,feed_dict = feed_dict)
                total_batch = total_batch + 1
                
                del x_batch
                del y_batch
                gc.collect()

                
                if total_batch - last_improved_batch == require_improved:
                    #验证集准确率长时间未提升，提前结束训练
                    print("No improvement for a long time, auto-stopping..................")
                    flag = True
                    break
            if flag:
                break
            

def test(test_dir):
    #获得训练和验证数据
    print("Loading test data...................")
    start_time = time.time()
    x_test,y_test = word_to_id(test_dir,vocab_size=config.vocab_size,max_length=config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:",time_dif)
    
    
    save_dir = "./best_model"
    save_path = os.path.join(save_dir,"best_validation")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #加载之前保存的模型
        saver = tf.train.Saver()
        saver.restore(sess,save_path=save_path)        
        
        print("Testing....................................")
        start_time = time.time()
        test_loss,test_acc = evaluate(sess,x_test,y_test)
        print("Test loss:%0.2f, Test accuracy:%0.2f" %(test_loss,test_acc))
        
        #获得预测结果，计算混淆矩阵等评测值。其实这一步和上一步在运算内容上也是有重复的
        y_true = np.argmax(y_test,1)
        y_pred = [0]*len(y_true)
        batch_size = 128
        batch_num = int((len(x_test)-1) / batch_size) + 1
        for i in range(batch_num):
            start_id = i*batch_size
            end_id = min((i+1)*batch_size,len(x_test))
            feed_dict = {
                    model.input_x:x_test[start_id:end_id],
                    model.dropout_prob:1.0
                    }
            y_pred[start_id:end_id] = sess.run(model.y_pred,feed_dict = feed_dict)
        y_pred = np.array(y_pred)
        
        #评估
        print("Precision,Recall and F1-score...................")
        category_id = build_category_id()
        categories = [k for k in category_id]   #类别
        print(metrics.classification_report(y_true,y_pred,target_names = categories))
        
        print("Confusion matrix...............................")
        print(metrics.confusion_matrix(y_true,y_pred))
        
        time_dif = get_time_dif(start_time)
        print("Time usage:",time_dif)
        




if __name__ == "__main__":
    config = RNNConfig()        
    model = TextRNN(config)
    
    train_dir = "./cnews_data/cnews.train.txt"
    val_dir = "./cnews_data/cnews.val.txt"
    test_dir = "./cnews_data/cnews.test.txt"
    
    train(train_dir,val_dir)

    test(test_dir)    
    









































