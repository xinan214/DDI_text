import numpy as np
import sys
import time
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from beam import Beam

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

class GraphAttentionLayer(nn.Module):
    # concat: 一个布尔值，指示是否将节点特征和注意力权重串联在一起作为输出
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(self.init_params(in_features, out_features), requires_grad=True)
        self.a1 = nn.Parameter(self.init_params(out_features, 1), requires_grad=True)
        self.a2 = nn.Parameter(self.init_params(out_features, 1), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha) # PyTorch 中的一个激活函数，它将输入的每个元素通过一个修正线性单元

# torch.FloatTensor(_in, _out) 创建了一个形状为(_in, _out)的张量，即一个大小为_in × _out的二维矩阵。然后，nn.init.xavier_uniform_ 对该张量进行初始化，
    # 使用Xavier均匀分布的方法来填充该矩阵，其中的权重值通过从均匀分布中采样并乘以根号0.01的方式进行缩放。最后，初始化后的张量作为参数 params 返回
    def init_params(self, _in, _out):
        params = nn.init.xavier_uniform_(torch.FloatTensor(_in, _out), gain=np.sqrt(0.01))
        return params

# 接收输入 _input 和邻接矩阵 adj
    def forward(self, _input, adj):
        # 将输入 _input 与权重矩阵 self.W 做矩阵乘法，得到节点特征 h
        h = torch.mm(_input, self.W)
        N = h.size()[0]

        # 分别通过两个线性变换 self.a1 和 self.a2 将节点特征 h 转换为注意力系数 f_1 和 f_2。
        # 通过线性变换将节点特征转换为注意力系数的目的是引入可学习的参数，使模型能够自动学习节点之间的关联程度
        # 通过 LeakyReLU 激活函数对 f_1 和 f_2 进行非线性激活，并将它们相加得到注意力系数矩阵 e
        f_1 = h @ self.a1
        f_2 = h @ self.a2
        # 通过 LeakyReLU 激活函数对 f_1 和 f_2 进行非线性激活，并将它们相加得到注意力系数矩阵 e
        # 这个非线性变换的目的是引入一种非线性的关系，使得模型可以更好地对节点之间的关联程度进行建模
        # f_2.transpose(0, 1)转置 α⊢g (vi, vj ) = LeakyReLU((a⊢g )T [W⊢g vi||W⊣g vj ])
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))  # node_num * node_num

        # 为了计算注意力权重，首先创建一个形状与 e 相同的全为负无穷大的张量 zero_vec。然后，
        # 使用 torch.where 函数根据邻接矩阵 adj 的值选择性地将 e 或 zero_vec 赋值给 attention，这样可以将无关节点的注意力权重设置为负无穷大，以忽略它们的影响
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 据邻接矩阵 adj 的值选择性地将注意力系数矩阵 e 中的值或负无穷大的张量 zero_vec 中的值赋值给注意力权重矩阵 attention
        # 对 attention 进行 softmax 归一化，并使用 dropout 随机地将一部分注意力权重设为零
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 最后，通过注意力权重矩阵 attention 将节点特征 h 进行加权求和，得到更新后的节点特征 h_prime
        h_prime = torch.matmul(attention, h)
        # if self.concat:
        #   return F.elu(h_prime)
        # else:
        #   return h_prime
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Encoder(nn.Module):
    ''' Graph Attention Embedding'''

    def __init__(self,
                 word_embedding,
                 node_embedding,
                 n_nodes,
                 n_features,
                 hidden_size,
                 dropout,
                 alpha, # 0.2
                 nheads, # 4
                 n_ctxs, # 2
                 ctx_attn, # 'sum_co'
                 integration):
        super(Encoder, self).__init__()

        self.word_embedding = word_embedding
        self.node_embedding = node_embedding
        self.n_nodes = n_nodes
        self.out_feature = hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_ctxs = n_ctxs # 上下文数目
        self.ctx_attn = ctx_attn # 上下文注意力机制类型
        self.integration = integration # 集成方法

        # Graph Encoder
        # 通过GraphAttentionLayer类创建了多个图注意力层，分别用于外部图编码和内部图编码。每个图注意力层都具有相同的输入特征维度、输出特征维度和其他超参数
        # concat: 一个布尔值，指示是否将节点特征和注意力权重串联在一起作为输出
        self.outer_attentions = [GraphAttentionLayer(n_features,
                                                     self.out_feature // nheads,
                                                     dropout,
                                                     alpha,
                                                     True) for _ in range(nheads)]
        self.outer_out = nn.Linear(self.out_feature, self.out_feature)

        self.inner_attentions = [GraphAttentionLayer(n_features,
                                                     self.out_feature // nheads,
                                                     dropout,
                                                     alpha,
                                                     True) for _ in range(nheads)]
        self.inner_out = nn.Linear(self.out_feature, self.out_feature) #用于下面的分类任务

        # add modules
        for i, attention in enumerate(self.outer_attentions):
            self.add_module('outer_attention_{}'.format(i), attention)

        for i, attention in enumerate(self.inner_attentions):
            self.add_module('inner_attention_{}'.format(i), attention)

        # Text Encoder
        self.in_encoder_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out_encoder_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.in_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.in_att = nn.Linear(self.hidden_size, 1)
        self.out_att = nn.Linear(self.hidden_size, 1)

        self.out_att_concat = nn.Linear(self.hidden_size * self.n_ctxs, self.hidden_size)
        self.in_att_concat = nn.Linear(self.hidden_size * self.n_ctxs, self.hidden_size)

        #  Gated Neural Fusion
        self.head_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tail_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.link_multi_view_cat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gen_multi_view_cat = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.link_multi_view_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.gen_multi_view_add = nn.Linear(self.hidden_size, self.hidden_size)

        self.multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)  # 只用一个门控
        self.link_multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)
        self.gen_multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)

    def graph_encoder(self, in_data, out_data):
        # in_nodes 和 out_nodes 分别表示输入和输出节点的 ID 列表
        # in_map_hs, in_map_ts 是什么 in_map_hs 和 in_map_ts，它们分别表示输入节点 ID 到索引的映射关系 用于将输入节点 ID 转换为对应的索引，从而得到对应的节点特征向量
        # inner_adj 和 outer_adj 分别表示输入和输出节点之间的连接关系，用于构建邻接矩阵
        in_nodes, inner_adj, in_map_hs, in_map_ts = in_data
        out_nodes, outer_adj, out_map_hs, out_map_ts = out_data

        out_nodes_emb = self.node_embedding(out_nodes)
        # print("out_nodes_emb:")
        # print(out_nodes_emb)
        in_nodes_emb = self.node_embedding(in_nodes)

        # [n, emb_size]
        x = F.dropout(out_nodes_emb, self.dropout, training=self.training)
        # 输入的是初始向量和权重矩阵
        # 将输入节点的特征矩阵 x 分别传入多个图注意力层 self.outer_attentions 进行处理，并将所有处理后的结果按列方向（dim=1）拼接起来，
        # 得到新的特征矩阵 out_emb，用于表示输入节点在外部图上的特征向量表示
        out_emb = torch.cat([att(x, outer_adj) for att in self.outer_attentions], dim=1)

        # [n, hidden_size]
        x = F.dropout(in_nodes_emb, self.dropout, training=self.training)
        in_emb = torch.cat([att(x, inner_adj) for att in self.outer_attentions], dim=1)

        in_head = in_emb[in_map_hs]
        in_tail = in_emb[in_map_ts]
        out_head = out_emb[out_map_hs]
        out_tail = out_emb[out_map_ts]

        head = F.elu(in_head + out_head)
        tail = F.elu(in_tail + out_tail)
        head = F.dropout(head, self.dropout, training=self.training)
        tail = F.dropout(tail, self.dropout, training=self.training)
        print("model.graph_encoder成功")
        return head, tail # head 和 tail 分别表示输入节点在外部图和内部图上的特征向量表示

    
    def gat_encoder(self, out_nodes, in_nodes, outer_adj, inner_adj):
        out_nodes_emb = self.node_embedding(out_nodes)
        in_nodes_emb = self.node_embedding(in_nodes)

        # [n, emb_size]
        x = F.dropout(out_nodes_emb, self.dropout, training=self.training)
        outer_x = torch.cat([att(x, outer_adj) for att in self.outer_attentions], dim=1)
        outer_x = F.dropout(outer_x, self.dropout, training=self.training)
        outer_x = self.outer_out(outer_x)

        # [n, hidden_size]
        x = F.dropout(in_nodes_emb, self.dropout, training=self.training)
        inner_x = torch.cat([att(x, inner_adj) for att in self.inner_attentions], dim=1)
        inner_x = F.dropout(inner_x, self.dropout, training=self.training)
        inner_x = self.inner_out(inner_x)
        print("model.gat_encoder成功")
        return outer_x, inner_x

    
    def text_encoder(self, in_ctx_data, out_ctx_data):
        in_ctx, in_ctx_lengths = in_ctx_data
        out_ctx, out_ctx_lengths = out_ctx_data

        in_input_embedded = self.word_embedding(in_ctx)
        out_input_embedded = self.word_embedding(out_ctx)
#in_input_embedded.transpose(1, 0): 对输入的嵌入序列进行了转置操作，将原始形状为(sequence_length, batch_size, embedding_size)的张量转换为(batch_size, sequence_length, embedding_size)的张量。
        # 这是因为pack_padded_sequence函数要求序列的维度为(sequence_length, batch_size, ...)。

#in_ctx_lengths: 这是一个包含了输入序列每个样本的有效长度（非填充部分的长度）的列表。这个列表的长度为batch_size，每个元素表示对应样本的有效长度。
        # 该参数用于告知pack_padded_sequence函数每个样本的实际长度，以便进行正确的填充处理
        padded_in_input_embedded = nn.utils.rnn.pack_padded_sequence(in_input_embedded.transpose(1, 0),
                                                                     in_ctx_lengths, #.cpu()
                                                                     enforce_sorted=False)
        padded_out_input_embedded = nn.utils.rnn.pack_padded_sequence(out_input_embedded.transpose(1, 0),
                                                                      out_ctx_lengths, #.cpu() 
                                                                      enforce_sorted=False)

# 使用GRU（门控循环单元）对填充处理后的输入序列进行编码
        in_output, in_hidden = self.in_encoder_gru(padded_in_input_embedded)
        out_output, out_hidden = self.out_encoder_gru(padded_out_input_embedded)

        if self.ctx_attn == 'concat':
            in_hidden = in_hidden.reshape([-1, self.hidden_size * self.n_ctxs])
            out_hidden = out_hidden.reshape([-1, self.hidden_size * self.n_ctxs])
            concat_in_hidden = self.in_att_concat(in_hidden)
            concat_out_hidden = self.out_att_concat(out_hidden)

            return concat_in_hidden, concat_out_hidden

        in_hidden = in_hidden.reshape([-1, self.n_ctxs, self.hidden_size])
        out_hidden = out_hidden.reshape([-1, self.n_ctxs, self.hidden_size])

        if self.ctx_attn == 'avg':
            avg_in_hidden = torch.mean(in_hidden, 1)
            avg_out_hidden = torch.mean(out_hidden, 1)
            return avg_out_hidden, avg_in_hidden

        elif self.ctx_attn == 'sum_co':
            attn_in2out = in_hidden.bmm(out_hidden.transpose(1, 2)).contiguous()
            attn_in2out = F.softmax(torch.sum(attn_in2out, 1), dim=1)

            attn_out2in = out_hidden.bmm(in_hidden.transpose(1, 2)).contiguous()
            attn_out2in = F.softmax(torch.sum(attn_out2in, 1), dim=1)

            out_hidden = attn_in2out.unsqueeze(1).bmm(out_hidden).squeeze()
            in_hidden = attn_out2in.unsqueeze(1).bmm(in_hidden).squeeze()
            return out_hidden, in_hidden

        elif self.ctx_attn == 'single':
            # single-attention
            in_hidden = in_hidden.reshape([-1, self.n_ctxs, self.hidden_size])  # .transpose(1,2)
            out_hidden = out_hidden.reshape([-1, self.n_ctxs, self.hidden_size])  # .transpose(1,2)

            in_attn = F.tanh(self.in_att(in_hidden))
            out_attn = F.tanh(self.out_att(out_hidden))

            in_attn = torch.softmax(in_attn, 1)
            out_attn = torch.softmax(out_attn, 1)

            attn_in_hidden = torch.bmm(in_attn.transpose(1, 2), in_hidden).squeeze()
            attn_out_hidden = torch.bmm(out_attn.transpose(1, 2), out_hidden).squeeze()
            print("model.text_encoder成功")
            return attn_out_hidden, attn_in_hidden

        '''
        in_output, _ = nn.utils.rnn.pad_packed_sequence(in_output)
        out_output, _ = nn.utils.rnn.pad_packed_sequence(out_output) 

        # co-attention
        in_output = in_output.transpose(0, 1)
        in_att = torch.unsqueeze(self.in_s(out_hidden.squeeze()), 2) 
        in_att_score = F.softmax(torch.bmm(in_output, in_att), dim=1)
        cross_in_hidden = torch.squeeze(torch.bmm(in_output.transpose(1,2), in_att_score))

        out_output = out_output.transpose(0, 1)
        out_att = torch.unsqueeze(self.out_s(in_hidden.squeeze()), 2) 
        out_att_score = F.softmax(torch.bmm(out_output, out_att), dim=1)
        cross_out_hidden = torch.squeeze(torch.bmm(out_output.transpose(1,2), out_att_score))

        return cross_in_hidden, cross_out_hidden
        '''
    

    def forward(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data):

        # print("hs.shape:",hs.shape) #([96])

        # Graph-view
        graph_head, graph_tail = self.graph_encoder(in_data, out_data)
        # Text-view    
        text_head, text_tail = self.text_encoder(in_ctx_data, out_ctx_data)

        if self.integration == 'concat':
            head = torch.cat([graph_head, text_head], dim=1)
            tail = torch.cat([graph_tail, text_tail], dim=1)

            link_head = self.link_multi_view_cat(head)
            link_tail = self.link_multi_view_cat(tail)

            gen_head = self.gen_multi_view_cat(head)
            gen_tail = self.gen_multi_view_cat(tail)

            return link_head, link_tail, gen_head, gen_tail

        elif self.integration == 'add':
            head = graph_head + text_head
            tail = graph_tail + text_tail

            link_head = self.link_multi_view_add(head)
            link_tail = self.link_multi_view_add(tail)

            gen_head = self.gen_multi_view_add(head)
            gen_tail = self.gen_multi_view_add(tail)
            return link_head, link_tail, gen_head, gen_tail

        # head = torch.cat([graph_head, text_head], dim = 1)
        # tail = torch.cat([graph_tail, text_tail], dim = 1)

        # Gated-Combination 
        # gate_head = torch.sigmoid(self.multi_view_gate(hs))
        # gate_tail = torch.sigmoid(self.multi_view_gate(ts))

        # head = gate_head * graph_head + (1-gate_head) * text_head
        # tail = gate_tail * graph_tail + (1-gate_tail) * text_tail
        # return head, tail

        # Gated-Combination 
        link_gate_head = torch.sigmoid(self.link_multi_view_gate(hs))
        link_gate_tail = torch.sigmoid(self.link_multi_view_gate(ts))
        link_head = link_gate_head * graph_head + (1 - link_gate_head) * text_head
        link_tail = link_gate_tail * graph_tail + (1 - link_gate_tail) * text_tail

        gen_gate_head = torch.sigmoid(self.gen_multi_view_gate(hs))
        gen_gate_tail = torch.sigmoid(self.gen_multi_view_gate(ts))
        gen_head = gen_gate_head * graph_head + (1 - gen_gate_head) * text_head
        gen_tail = gen_gate_tail * graph_tail + (1 - gen_gate_tail) * text_tail

        return link_head, link_tail, gen_head, gen_tail


class Decoder(nn.Module):
    def __init__(self, n_words, n_nodes, max_len, opt):
        super(Decoder, self).__init__()
        self.n_words = n_words # 9860
        self.n_nodes = n_nodes # 101
        self.max_len = max_len # 191
        self.n_features = opt.n_features # 128
        self.hidden_size = opt.hidden_size # 128
        self.dropout = opt.dropout # 0.6
        self.beam_size = opt.beam_size # 2
        self.task = opt.task # multi
        self.lamb = opt.lamb # 0.05
        self.gpu = opt.gpu # False
        self.integration = opt.integration # ''


        self.node_embedding = nn.Embedding(n_nodes, self.n_features) # torch.Size([101, 128])
        self.word_embedding = nn.Embedding(self.n_words, self.hidden_size) # torch.Size([9860, 128])

        self.linear = nn.Linear(self.n_features, 114) # 还有一种类型是不反应（负样本）

        # Graph Encoder
        self.encoder = Encoder(word_embedding=self.word_embedding,
                               node_embedding=self.node_embedding,
                               n_nodes=self.n_nodes,
                               n_features=self.n_features,
                               hidden_size=self.hidden_size,
                               dropout=opt.dropout,
                               alpha=opt.alpha,
                               nheads=opt.nheads,
                               n_ctxs=opt.n_ctxs,
                               ctx_attn=opt.ctx_attn,
                               integration=self.integration)

        # Text Decoder
        self.head_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        # 激活函数，它将输入的每个元素通过一个双曲正切函数进行映射
        self.tanh = nn.Tanh()
        self.decode_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.V = nn.Linear(self.hidden_size, self.n_words)

        # Loss
        # 这个函数会自动将输出先经过一个softmax函数，然后将softmax输出的结果作为输入给交叉熵损失函数
        self.link_loss = nn.CrossEntropyLoss() # self.link_loss = nn.BCELoss()
        self.gen_loss = nn.CrossEntropyLoss()

    def init_params(self, _in, _out):
        params = nn.init.xavier_uniform_(torch.FloatTensor(_in, _out), gain=np.sqrt(2.0))
        return params

    def decode(self, head, tail, w_input, w_output, input_lengths):
        [batch_size, max_length] = w_input.shape

#         hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0) 
#         input_embedded = self.word_embedding(w_input)
#         side_input = torch.cat([head, tail], dim = 1).unsqueeze(1).repeat(1, max_length, 1)
#         input_embedded = self.side_test(torch.cat((side_input, input_embedded), dim=2))
# 首先对输入序列进行词嵌入操作。然后使用头节点和尾节点生成初始隐藏状态（init_hidden）
        input_embedded = self.word_embedding(w_input)
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)

        init_hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(1)
        # init_hidden = self.hidden_test(torch.cat([head, tail], dim = 1)).unsqueeze(1)
        init_word = F.softmax(self.V(init_hidden), dim=-1).topk(1)[1].squeeze()

        # init_word = self.V(init_hidden).topk(1)[1].squeeze()
        init_input_embedded = self.word_embedding(init_word).unsqueeze(1)
        input_embedded = torch.cat([init_input_embedded, input_embedded], dim=1)

        input_embedded = nn.utils.rnn.pack_padded_sequence(input_embedded,
                                                           input_lengths, #.cpu()
                                                           enforce_sorted=False,
                                                           batch_first=True)
# 通过GRU（decode_gru）对输入序列进行解码，得到输出和最后的隐藏状态
        output, hidden = self.decode_gru(input_embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

#         weighted_pvocab = self.V(output)
#         loss = self.gen_loss(weighted_pvocab.view(-1, self.n_words), w_output.view(-1))
#         return loss
        weighted_pvocab = F.softmax(self.V(output), dim=-1)
        target_id = w_output.unsqueeze(2)
        output = weighted_pvocab.gather(2, target_id).add_(sys.float_info.epsilon)
        target_mask_0 = target_id.ne(0).detach()
        loss = (output.log().mul(-1) * target_mask_0.float()).squeeze().sum(1).div(max_length)
        return loss.mean()
    

    def forward(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data, w_input, w_output, input_lengths, label):
        label = torch.tensor(label)
        # print("model.label:",label.shape) # 96
        link_head, link_tail, gen_head, gen_tail = self.encoder(hs,
                                                                ts,
                                                                in_data,
                                                                out_data,
                                                                in_ctx_data,
                                                                out_ctx_data)

        # print("link_head.size:",link_head.shape) #([96])
        # link prediciton
        head, tail = link_head, link_tail
        pos_num = w_input.shape[0]
        neg_num = link_head.shape[0] - pos_num

        num_classes = 114  # 标签有113个类别 负样本为0
        labels = torch.nn.functional.one_hot(label, num_classes=num_classes).float()
        # print("model.labels:", labels.shape) # 96,114
        x = torch.mul(head, tail)
        x = self.linear(x)
        scores = torch.nn.functional.softmax(x, dim=1)
        if self.gpu: labels = labels.cuda()
        link_loss = self.link_loss(scores, labels)
        print("loss:",link_loss)

        # context generation
        head, tail = gen_head, gen_tail
        pos_head = head[:pos_num, :]
        pos_tail = tail[:pos_num, :]
        gen_loss = self.decode(pos_head, pos_tail, w_input, w_output, input_lengths)

        if self.task == 'link':
            loss = self.lamb *link_loss
        elif self.task == 'gen':
            loss = gen_loss
        else:
            loss = self.lamb *link_loss + gen_loss
            # loss = link_loss + gen_loss
        return scores, link_loss, gen_loss, loss

# 用于对编码器进行评估，接收头节点（hs）、尾节点（ts）、输入数据（in_data）、输出数据（out_data）、输入上下文数据（in_ctx_data）和输出上下文数据（out_ctx_data）作为输入，
    # 并返回链接预测的头节点（link_head）、尾节点（link_tail）、生成预测的头节点（gen_head）和尾节点（gen_tail）
    def evaluation_encoder(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data):
        link_head, link_tail, gen_head, gen_tail = self.encoder(hs, ts,
                                                                in_data,
                                                                out_data,
                                                                in_ctx_data,
                                                                out_ctx_data)
        return link_head, link_tail, gen_head, gen_tail

# evaluate_decode 函数：用于进行解码评估，接收头节点（head）和尾节点（tail）作为输入。首先根据头节点和尾节点生成初始隐藏状态（hidden）。
    # 然后，通过循环解码过程，使用当前隐藏状态（hidden）和输出层（V）生成下一个单词的概率分布，选择概率最高的单词作为解码结果。最后返回解码结果
    def evaluate_decode(self, head, tail):
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)
        output = hidden

        decoded_outputs = list()
        for _step in range(self.max_len):
            weighted_pvocab = F.softmax(self.V(output), dim=-1).squeeze()
            symbols = weighted_pvocab.topk(1)[1]
            if (symbols.detach().item() == EOS_TOKEN):
                break
            decoded_outputs.append(symbols.detach().item())
            w_input = symbols
            print("word_embedding(w_input):",self.word_embedding(w_input))
            word_embedded = self.word_embedding(w_input).view(1, -1, self.hidden_size)
            # word_embedded = self.side_test(torch.cat((side_input, word_embedded), dim=2))  
            output, hidden = self.decode_gru(word_embedded, hidden)

        return decoded_outputs

# 用于进行束搜索解码评估，接收头节点（head）和尾节点（tail）作为输入。首先根据头节点和尾节点生成初始隐藏状态（hidden）。
    # 然后，进行束搜索解码过程，使用当前隐藏状态和输出层生成下一个单词的概率分布，并根据概率分布选择概率最高的若干个候选单词。根据候选单词更新束搜索器的状态，直到达到最大长度或满足停止条件。
    # 最后返回束搜索器的最佳解码结果
    def evaluate_beam_decode(self, head, tail):
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)
        output = hidden

        # Initial beam search
        weighted_pvocab = F.softmax(self.V(output), dim=-1).squeeze()
        init_symbols = weighted_pvocab.topk(1)[1].repeat(self.beam_size)
        beam = Beam(self.beam_size, init_symbols, gpu=self.gpu)

        # Generation
        hidden = hidden.repeat(1, self.beam_size, 1)
        decoded_outputs = list()
        for _step in range(self.max_len):
            input = beam.get_current_state()
            word_embedded = self.word_embedding(input).view(-1, 1, self.hidden_size)
            output, hidden = self.decode_gru(word_embedded, hidden)
            word_lk = F.softmax(self.V(output.transpose(1, 0)), dim=-1)

            if beam.advance(word_lk.data): break

            hidden.data.copy_(hidden.data.index_select(1, beam.get_current_origin()))

        scores, ks = beam.sort_best()
        return beam.get_hyp(ks[0])
