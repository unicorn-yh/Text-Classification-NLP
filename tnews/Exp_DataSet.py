import os
import json
import numpy as np
import torch
import jieba
from torch.utils.data import TensorDataset
from gensim.models import KeyedVectors

#-----------------------------------------------------begin-----------------------------------------------------#
# 全局变量
class Config():
    """配置参数"""
    def __init__(self):
        self.vocab_size = 0
        self.train_count = 500   # max train count = 53360
        self.test_count  = 100   # max test  count = 10000
        self.valid_count = 100   # max valid count = 10000
        self.embedding_dim = 0
        self.saveEmbedding = True
        self.max_sent_len = 0

config = Config()
#------------------------------------------------------end------------------------------------------------------#

class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {}
        self.tkn2word = []

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的 tokens 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行简化，只需调用预训练 tokenizer 即可将文本的数据全部转为 tokens 的数据。
    '''
    def __init__(self, path, max_sent_len):
        self.dictionary = Dictionary(path)

        self.max_sent_len = max_sent_len

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)
        

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embeddings 的映射矩阵  

        get_embedding = False
        file_path = 'embedding/embedding_matrix_'+str(config.train_count)+'.txt'

        if os.path.exists(file_path):
            if os.path.getsize(file_path) > 0:
                self.embedding_matrix = np.loadtxt(file_path)
            else:
                get_embedding = True
        else:
            get_embedding = True
        
        if get_embedding == config.saveEmbedding:
            id2word = self.dictionary.tkn2word
            word2vec_model = KeyedVectors.load_word2vec_format('lib/sgns.baidubaike.bigram-char',binary=False)
            self.embedding_matrix = []
            oov_words = []
            for word in id2word:  
                try:
                    self.embedding_matrix.append(word2vec_model[word])
                except:
                    oov_words.append(word)
                    self.embedding_matrix.append(np.zeros(300,dtype=np.float32))
            np.savetxt(file_path,self.embedding_matrix)
            self.embedding_matrix = np.loadtxt(file_path)
            np.savetxt('embedding/oov_words.txt',oov_words,fmt='%s',encoding="utf-8")
        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_sent):
        '''
        padding: 将一个 sentence 补 0 至预设的最大句长 self.max_sent_len
        '''
        if len(origin_sent) > self.max_sent_len:
            return origin_sent[:self.max_sent_len]
        else:
            return origin_sent + [0 for _ in range(self.max_sent_len-len(origin_sent))]

    def tokenize(self, path, test_mode=False):
        '''
        将数据集的每一个 sent 都转化成对应的 tokens. 
        '''
        with open(path, 'r', encoding='utf8') as f:
            if "dev" in path:
                count = config.valid_count
            elif "test" in path:
                count = config.test_count
            elif "train" in path:
                count = config.train_count
            index = 0
            data_ls = []
            if test_mode:
                idss = []
                for line in f:
                    #-----------------------------------------------------begin-----------------------------------------------------#
                    # 修改测试样本数量
                    index += 1
                    if index == count+1:
                        break
                    #------------------------------------------------------end------------------------------------------------------#

                    one_data = json.loads(line)  # 读取一条数据
                    sent = one_data['sentence']

                    #-----------------------------------------------------begin-----------------------------------------------------#
                    # 若采用预训练的 embedding, 可以在此处对 sent 分词操作
                    sent = list(jieba.cut(sent, cut_all=True))
                    #------------------------------------------------------end------------------------------------------------------#
                    # 向词典中添加词
                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    idss.append(self.pad(ids))

                idss = torch.tensor(np.array(idss))

                return TensorDataset(idss)

            else:
                idss = []
                labels = []
                
                for line in f:
                    #-----------------------------------------------------begin-----------------------------------------------------#
                    # 修改训练样本数量
                    index += 1
                    if index == count+1:
                        break
                    #------------------------------------------------------end------------------------------------------------------#
                   
                    one_data = json.loads(line)  # 读取一条数据

                    sent = one_data['sentence']
                    label = one_data['label']

                    #-----------------------------------------------------begin-----------------------------------------------------#
                    # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                    sent = list(jieba.cut(sent, cut_all=True))
                    #------------------------------------------------------end------------------------------------------------------#
                    
                    # 向词典中添加词
                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    idss.append(self.pad(ids))
                    labels.append(self.dictionary.label2idx[label])


                idss = torch.tensor(np.array(idss))      #词向量
                labels = torch.tensor(np.array(labels)).long()    #标签

                return TensorDataset(idss,labels)
