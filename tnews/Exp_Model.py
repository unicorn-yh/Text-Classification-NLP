import torch.nn as nn
import torch as torch
import math
import torch.nn.functional as F
from Exp_DataSet import config
import copy
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

#-----------------------------------------------------begin-----------------------------------------------------#
# 类别数
num_classes = 15
#------------------------------------------------------end------------------------------------------------------#

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ninp=300, ntoken=150, nhid=1200, nhead=10, nlayers=6, dropout=0.2, embedding_weight=None, name=None):
        super(Transformer_model, self).__init__()
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计词嵌入层
        if "ATTENTION" in name:
            self.attention = True
        else:
            self.attention = False
        self.name = name
        print(self.name.center(105,'='))
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=ninp, padding_idx=vocab_size-1)
        self.embed.weight.data.copy_(torch.from_numpy(embedding_weight))
        #------------------------------------------------------end------------------------------------------------------#
        
        self.pos_encoder = PositionalEncoding(d_model=ninp, max_len=ntoken)

        if self.attention == False:
            self.encode_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 请自行设计对 transformer 隐藏层数据的处理和选择方法
    # 请自行设计分类器
        
        elif self.attention == True:  # 使用 Multi-head Attention

            # Multi-Head Attention
            self.nlayers = nlayers
            self.nhead = nhead
            assert ninp % nhead == 0 
            self.dim_head = ninp // self.nhead
            self.fc_Q = nn.Linear(ninp, nhead * self.dim_head)
            self.fc_K = nn.Linear(ninp, nhead * self.dim_head)
            self.fc_V = nn.Linear(ninp, nhead * self.dim_head)
            self.fc = nn.Linear(nhead * self.dim_head, ninp)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(ninp)

            # Position Wise Feed Forward
            self.fc1 = nn.Linear(ninp, nhid)
            self.fc2 = nn.Linear(nhid, ninp)

        self.fc_out = nn.Linear(ninp*config.max_sent_len, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def Scaled_Dot_Product_Attention(self, Q, K, V, d_k=None):
        d_k = Q.size(-1) ** -0.5  # 缩放因子
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if d_k:
            attention = attention * d_k
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

    def Position_Wise_Feed_Forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

    #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
              
        if self.attention == False:
            x = self.embed(x)                # [batch_size, sentence_length, embedding_size]   64 30 300
            x = x.permute(1, 0, 2)           # [sentence_length, batch_size, embedding_size]   30 64 300 
            x = self.pos_encoder(x)          # [sentence_length, batch_size, embedding_size]   30 64 300
            x = self.transformer_encoder(x)  # [sentence_length, batch_size, embedding_size]   30 64 300                 
            x = x.permute(1, 0, 2)           # [batch_size, sentence_length, embedding_size]   64 30 300
    
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类

        elif self.attention == True:   
            x = self.embed(x)
            x = self.pos_encoder(x)

            for i in range(12):  
                batch_size = x.size(0)
                Q = self.fc_Q(x)
                K = self.fc_K(x)
                V = self.fc_V(x)
                Q = Q.view(batch_size * self.nhead, -1, self.dim_head)
                K = K.view(batch_size * self.nhead, -1, self.dim_head)
                V = V.view(batch_size * self.nhead, -1, self.dim_head)

                context = self.Scaled_Dot_Product_Attention(Q, K, V)

                context = context.view(batch_size, -1, self.dim_head * self.nhead)
                out = self.fc(context)
                out = self.dropout(out)
                out = out + x  # 残差连接
                out = self.layer_norm(out)

                x = self.Position_Wise_Feed_Forward(out)

        x = x.reshape(x.size(0), -1)
        x = self.fc_out(x)
        x = self.softmax(x)
        #------------------------------------------------------end------------------------------------------------------#

        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ninp=300, ntoken=150,  nhid=150, nlayers=4, dropout=0.2, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 自行设计词嵌入层
        self.name = "Bi-LSTM"
        print(self.name.center(105,'='))
        self.embed = nn.Embedding(vocab_size, ninp, padding_idx=vocab_size - 1)
        self.embed.weight.data.copy_(torch.from_numpy(embedding_weight))
        #------------------------------------------------------end------------------------------------------------------#

        self.lstm = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        # 请自行设计分类器

        self.num_layers = nlayers
        self.num_directions = 2
        self.hidden_size = nhid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(nhid, num_classes)
        self.act_func = nn.Softmax(dim=1)
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)        # [batch_size, sentence_length, embedding_size]      64 30 100
        #x = self.lstm(x)[0]
        #x = self.dropout(x)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        
        batch_size = x.size(0)  #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(0)获取当前数据实际的batch

        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)

        #out[max_sent_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out    #[batch_size, max_sent_len, hidden_size] 64 30 300

        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)   #[batch_size, num_layers * num_directions, hidden_size]
        h_n = torch.sum(h_n, dim=1)  #[batch_size, 1, hidden_size]
        h_n = h_n.squeeze(dim=1)     #[batch_size, hidden_size]

        attention_w = self.attention_weights_layer(h_n)  #[batch_size, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch_size, 1, hidden_size]

        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch_size, 1, max_sent_len]
        softmax_w = F.softmax(attention_context, dim=-1)                 #[batch_size, 1, max_sent_len]  权重归一化

        x = torch.bmm(softmax_w, out)  #[batch_size, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch_size, hidden_size]
        x = self.fc(x)
        x = self.act_func(x)
        #------------------------------------------------------end------------------------------------------------------#
        
        #x = self.classifier(x)
        return x

      
#-----------------------------------------------------begin-----------------------------------------------------#
# BERT 预训练模型

'''BERT ATT'''
class BertATT(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    """

    def __init__(self, config, num_labels=15):
        super(BertATT, self).__init__(config)
        self.name = 'BERT-ATT'
        print(self.name.center(105,'='))
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, max_sent_len, bert_dim=768]

        score = torch.tanh(torch.matmul(encoded_layers, self.W_w))
        # score: [batch_size, max_sent_len, bert_dim]

        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, max_sent_len, 1]

        scored_x = encoded_layers * attention_weights
        # scored_x : [batch_size, max_sent_len, bert_dim]

        feat = torch.sum(scored_x, dim=1)
        # feat: [batch_size, bert_dim=768]
        logits = self.classifier(feat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



'''BERT LSTM'''

class BertLSTM(BertPreTrainedModel):

    def __init__(self, config, rnn_hidden_size, num_layers, bidirectional, dropout, num_labels=num_classes):
        super(BertLSTM, self).__init__(config)
        self.name = 'BERT-LSTM'
        print(self.name.center(105,'='))
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


'''BERT-CNN'''

class BertCNN(BertPreTrainedModel):
      
    def __init__(self, config, n_filters, filter_sizes, num_labels=15):
        super(BertCNN, self).__init__(config)
        self.name = 'BERT-CNN'
        print(self.name.center(105,'='))
        self.num_labels = num_classes
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        
        encoded_layers = self.dropout(encoded_layers)

        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim=768, seq_len]

        conved = self.convs(encoded_layers)
        # conved 是一个列表， conved[0]: [batch_size, filter_num, *]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # pooled 是一个列表， pooled[0]: [batch_size, filter_num]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [batch_size, filter_num * len(filter_sizes)]

        logits = self.classifier(cat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]
#------------------------------------------------------end------------------------------------------------------#

