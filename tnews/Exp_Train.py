import torch
import torch.nn as nn
import time
from datetime import datetime
import numpy as np
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus, config
from Exp_Model import BiLSTM_model, Transformer_model, BertATT, BertLSTM, BertCNN
import bertatt_args, bertcnn_args, bertlstm_args

#-----------------------------------------------------begin-----------------------------------------------------#
# 设置想要训练的模型
#MODEL = "TRANSFORMER"
#MODEL = "TRANSFORMER-ATTENTION"
#MODEL = "Bi-LSTM"
#MODEL = "BERT-ATT"
MODEL = "BERT-LSTM"
#MODEL = "BERT-CNN"
output = []
#------------------------------------------------------end------------------------------------------------------#


def train():
    '''
    完成一个 epoch 的训练
    '''
    sum_true = 0
    sum_loss = 0.0

    max_valid_acc = 0

    model.train()
    index = 0
    total_data = len(data_loader_train)
    for data in data_loader_train:
              
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 可视化训练过程
        index += 1
        print('Training batch {}/{}'.format(index,total_data),end='\r')
        #------------------------------------------------------end------------------------------------------------------#
        
        # 选取对应批次数据的输入和标签
        batch_x, batch_y = data[0].to(device), data[1].to(device)

        # 模型预测
        y_hat = model(batch_x)
        loss = loss_function(y_hat, batch_y)

        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数

        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == batch_y).float()
        sum_loss += loss.item()

    train_acc = sum_true / dataset.train.__len__()
    train_loss = sum_loss / (dataset.train.__len__() / batch_size)

    valid_acc = valid()

    if valid_acc > max_valid_acc:
        torch.save(model, "checkpoint.pt")

    print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) }")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 把输出结果存储进文档
    output.append("epoch: "+str(epoch)+", train loss: "+str(round(float(train_loss),4))+", train accuracy: "+str(round(float(train_acc)*100,2))+"%, valid accuracy: "+str(round(float(valid_acc)*100,2))+"%, time: "+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    np.savetxt("result/"+model.name+"_"+str(config.train_count)+"_output.txt",output,fmt='%s')
    #------------------------------------------------------end------------------------------------------------------#
    


def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    sum_true = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader_valid:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            
            y_hat = model(batch_x)

            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            sum_true += torch.sum(y_hat == batch_y).float()

        return sum_true / dataset.valid.__len__()


def predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    results = []

    model = torch.load('checkpoint.pt').to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader_test:
            batch_x = data[0].to(device)

            y_hat = model(batch_x)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            results += y_hat.tolist()

    # 写入文件
    with open("predict.txt", "w") as f:
        for label_idx in results:
            label = dataset.dictionary.idx2label[label_idx][1]
            f.write(label+"\n")


if __name__ == '__main__':
    dataset_folder = 'data/tnews_public'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 100
    max_sent_len = 30
    batch_size = 64
    epochs = 20
    lr = 5e-5
    #------------------------------------------------------end------------------------------------------------------#

    dataset = Corpus(dataset_folder, max_sent_len)

    #-----------------------------------------------------begin-----------------------------------------------------#
    #
    vocab_size = len(dataset.embedding_matrix)   #
    config.vocab_size = vocab_size
    config.max_sent_len = max_sent_len
    #------------------------------------------------------end------------------------------------------------------#

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    # 设置模型

    output_dir = "result/"
    cache_dir = ".bert_cache"
    log_dir = ".bert_log/"
    bert_vocab_file = "lib/pretrain_bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "lib/pretrain_bert"

    if "TRANSFORMER" in MODEL:
        model = Transformer_model(vocab_size,embedding_weight=dataset.embedding_matrix,name=MODEL).to(device)  

    elif MODEL == "Bi-LSTM":
        model = BiLSTM_model(vocab_size,embedding_weight=dataset.embedding_matrix).to(device)

    elif MODEL == "BERT-ATT":
        bertconfig = bertatt_args.get_args(dataset_folder, output_dir, cache_dir,bert_vocab_file, bert_model_dir, log_dir)
        model = BertATT.from_pretrained(bertconfig.bert_model_dir).to(device)

    elif MODEL == "BERT-LSTM":
        bertconfig = bertlstm_args.get_args(dataset_folder, output_dir, cache_dir,bert_vocab_file, bert_model_dir, log_dir)
        model = BertLSTM.from_pretrained(bertconfig.bert_model_dir, rnn_hidden_size=bertconfig.hidden_size, num_layers=bertconfig.num_layers, bidirectional=bertconfig.bidirectional, dropout=bertconfig.dropout).to(device)

    elif MODEL == "BERT-CNN":
        bertconfig = bertcnn_args.get_args(dataset_folder, output_dir, cache_dir,bert_vocab_file, bert_model_dir, log_dir)
        model = BertCNN.from_pretrained(bertconfig.bert_model_dir, n_filters=bertconfig.filter_num, filter_sizes=tuple(int(num) for num in bertconfig.filter_sizes.split(' '))).to(device)
    #------------------------------------------------------end------------------------------------------------------#

    loss_function = nn.CrossEntropyLoss()                                       # 设置损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # 设置优化器

    # 进行训练
    for epoch in range(epochs):
        train()

    # 对测试集进行预测
    predict()
