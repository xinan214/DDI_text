import time
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import _pickle as cPickle
import pandas as pd

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


def compute_bleu(references, candidates):
    references = [[item] for item in references]
    smooth_func = SmoothingFunction().method0  # method0-method7
    score1 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0,))
    score2 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/2,)*2)
    score3 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/3,)*3)
    score4 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/4,)*4)
    return score1, score2, score3, score4


class Loader():
    def __init__(self, dataset, gpu):
        self.word2idx = {}
        self.idx2word = {0: "SOS", 
                         1: "EOS", 
                         2: "PAD"}
        self.n_words = len(self.idx2word) 
        self.G_out = nx.DiGraph() # 按照引用其他论文构造图
        self.G_in = nx.DiGraph() # 按照被引用论文构造图
        
        self.train_data, self.test_data = self.load_data(dataset)
        self.cons_vocabulary(self.train_data)
        self.cons_vocabulary(self.test_data)
        self.cons_graph(self.train_data)
        self.gpu = gpu
       

    def load_data(self, dataset):
        path = 'data/{}.pkl'.format(dataset)
        with open(path, 'rb') as fo:
            data = pickle.load(fo)

        train_data = data['train_data']
        # print("train.shape:")
        # print(train_data.shape) # 175869
        test_data = data['test_data']
        train_data = train_data.iloc[1:].reset_index(drop=True)
        test_data = test_data.iloc[1:].reset_index(drop=True)
        # train_data, test_data = pickle.load(open(path, 'rb'))

        train_data["num_of_words"] = train_data["num_of_words"].astype(np.int32)
        test_data["num_of_words"] = test_data["num_of_words"].astype(np.int32)

        # print("训练集数据：", train_data) # 一个batch中 1000 * 4
        # print("测试集数据：", test_data) # 一个batch中100*4
        self.max_len = max(train_data["num_of_words"].max(),test_data["num_of_words"].max())
        print("max_len:")
        print(self.max_len)
        # self.n_nodes = max(train_data["Drug A"].max(), train_data["Drug B"].max())+1
        drug_a_b = pd.concat([train_data['Drug A'], train_data['Drug B']])
        all_nodes = drug_a_b.unique()
        # 获取节点数
        self.n_nodes = len(all_nodes)
        # self.n_nodes = len(set(train_data["Drug A"]).union(set(train_data["Drug B"])))
        print("n_nodes:")
        print(self.n_nodes)
        print("data_load_over!")

        return train_data, test_data

    
    def build_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1  
            
            
    def cons_vocabulary(self, data):
        for index, row in data.iterrows():
            for word in row["DDI sentence"].split():
                self.build_word(word)
        self.build_word("PAD")
        # print("utils.cons_vocabulary正常")
        
        
    # def cons_graph(self, data):
    #     for p1, p2, c, _, *_ in data.values: # 多label
    #         self.G_out.add_edge(p1, p2, **{'DDI sentence': c})
    #         self.G_in.add_edge(p2, p1, **{'DDI sentence': c})
    #     print("utils.cons_graph正常")
    def cons_graph(self, data):
        for p1, p2, c, _, *_ in data.values:
            if p1 not in self.G_out:
                self.G_out.add_node(p1)
            if p2 not in self.G_out:
                self.G_out.add_node(p2)
            if p1 not in self.G_in:
                self.G_in.add_node(p1)
            if p2 not in self.G_in:
                self.G_in.add_node(p2)
            self.G_out.add_edge(p1, p2, **{'DDI sentence': c})
            self.G_in.add_edge(p2, p1, **{'DDI sentence': c})
        print("G_in has {} nodes and {} edges".format(self.G_in.number_of_nodes(), self.G_in.number_of_edges()))
        print("G_out has {} nodes and {} edges".format(self.G_out.number_of_nodes(), self.G_out.number_of_edges()))
        # print("utils.cons_graph正常")

    def subgraph(self, idx):
        # in_graph
        in_sub_nodes = set()
        for node in idx:
            in_sub_nodes.add(node)
            neibors = [n for n in self.G_in.neighbors(node) if n in self.G_in.nodes()]  # 仅选择存在于原始图中的节点
            in_sub_nodes = in_sub_nodes.union(set(neibors))
        in_sub_graph = nx.subgraph(self.G_in, list(in_sub_nodes))

        # out_graph
        out_sub_nodes = set()
        for node in idx:
            out_sub_nodes.add(node)
            neibors = [n for n in self.G_out.neighbors(node) if n in self.G_out.nodes()]  # 仅选择存在于原始图中的节点
            out_sub_nodes = out_sub_nodes.union(set(neibors))
        out_sub_graph = nx.subgraph(self.G_out, list(out_sub_nodes))
        # print("utils.subgraph正常")
        return in_sub_graph, out_sub_graph

    # def subgraph(self, idx):  # 查找一个节点的外部图和内部图
    #     # in_graph
    #     in_sub_nodes = set()
    #     for node in (idx):
    #         in_sub_nodes.add(node)
    #         neibors = self.G_in.neighbors(node)
    #         in_sub_nodes = in_sub_nodes.union(set(neibors))
    #     in_sub_graph = nx.subgraph(self.G_in, list(in_sub_nodes))
    #
    #     # out_graph
    #     out_sub_nodes = set()
    #     for node in (idx):
    #         out_sub_nodes.add(node)
    #         neibors = self.G_out.neighbors(node)
    #         out_sub_nodes = out_sub_nodes.union(set(neibors))
    #     out_sub_graph = nx.subgraph(self.G_out, list(out_sub_nodes))
    #     print("utils.subgraph正常")
    #     return in_sub_graph, out_sub_graph
    
    
    def graph_subtensor(self, hs, ts):         
        in_sub_graph, out_sub_graph = self.subgraph(hs+ts)
        
        in_nodes = list(in_sub_graph.nodes())

        # print("in_nodes type:")
        # print(type(in_nodes))
        # print(in_nodes) #[55, 0, 53, 24, 89, 56, 86, 29, 47, 70, 1, 16, 76, 54, 60, 3, 28, 37, 92, 8, 2, 75, 34, 67, 88, 68, 39, 32, 42, 98, 51, 5, 6, 15, 23, 48, 4, 22, 20, 44, 83, 96, 95, 90, 12, 61, 41, 62, 87, 50, 49, 94, 27, 59, 81, 36, 7, 21, 19, 73, 64, 72, 91, 63, 66, 97, 80, 9, 43, 31, 71, 10, 65, 52, 13, 11, 85, 40, 14, 99, 82, 38, 35, 84, 93, 46, 79, 33, 26, 30, 57, 17, 18, 25, 58, 77, 78, 74, 69, 45]
        # in_adj 是一个邻接矩阵，表示子图中节点之间的连接关系，它是一个 NumPy 数组。in_adj 的形状是 (num_in_nodes, num_in_nodes)
        in_adj =  np.array(nx.adjacency_matrix(in_sub_graph).todense())
        out_nodes = list(out_sub_graph.nodes())        
        out_adj = np.array(nx.adjacency_matrix(out_sub_graph).todense())
     
        # node mapping 
        in_node2idx = {n:i for i, n in enumerate(in_nodes)} # 键值对
        out_node2idx = {n:i for i, n in enumerate(out_nodes)}

        # in_map_hs 和 in_map_ts 是头实体和尾实体在子图中的索引，分别是 hs 和 ts 在 in_nodes 中的索引
        in_map_hs = [in_node2idx[i] for i in hs] # 形状为 [num_heads] 的一维长整型张量，表示子图中的头实体在节点列表中的索引
        in_map_ts = [in_node2idx[i] for i in ts]
        out_map_hs = [out_node2idx[i] for i in hs]
        out_map_ts = [out_node2idx[i] for i in ts]
        
        # convert to tensors
        # in_data：一个列表，表示子图中的入边信息。它是一个包含四个 PyTorch 张量的列表，分别表示子图中的节点、邻接矩阵、头实体的索引、尾实体的索引
        in_data = [torch.LongTensor(in_nodes), torch.FloatTensor(in_adj),
                   torch.LongTensor(in_map_hs), torch.LongTensor(in_map_ts)]
        # print("in_data中四个张量的形状：")
        # for x in in_data:
        #     print(x.shape)
        #     # torch.Size([100])
        #     # torch.Size([100, 100])
        #     # torch.Size([96])
        #     # torch.Size([96])
        out_data = [torch.LongTensor(out_nodes), torch.FloatTensor(out_adj),
                   torch.LongTensor(out_map_hs), torch.LongTensor(out_map_ts)]
        
        if self.gpu:
            hs = torch.LongTensor(hs).cuda()
            ts = torch.LongTensor(ts).cuda()
            in_data = [x.cuda() for x in in_data]
            out_data = [x.cuda() for x in out_data]
        else:
            hs = torch.LongTensor(hs)
            ts = torch.LongTensor(ts)
        # print("utils.graph_subtensor正常")
        return hs, ts, in_data, out_data
        
        
    # 在调用 text_subtensor 函数时输入的数据中包含了不存在于原始图中的节点
    def text_subtensor(self, in_ctx, t_lens, out_ctx, h_lens): 
        in_ctx_data = [nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in in_ctx], 
                            batch_first=True, padding_value = PAD_TOKEN), torch.LongTensor(t_lens)]
        
        out_ctx_data = [nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in out_ctx], 
                            batch_first=True, padding_value = PAD_TOKEN), torch.LongTensor(h_lens)]
        
        if self.gpu:
            in_ctx_data = [x.cuda() for x in in_ctx_data]
            out_ctx_data = [x.cuda() for x in out_ctx_data]
        # print("utils.text_subtensor正常")
        return in_ctx_data, out_ctx_data

 # w_input, w_output两个形状为 [batch_size, max_seq_len] 的长整型张量，分别表示一个 batch 中的单词序列和单词序列标签
    def gen_subtensor(self, w_input, w_output, n_of_words):
        # 这行代码使用了 PyTorch 中的 pad_sequence 函数，将输入的多个序列按照最长的序列进行填充，使它们的长度都相等
        w_input = nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in w_input], 
                                            batch_first=True, padding_value = PAD_TOKEN)
        w_output = nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in w_output], 
                                             batch_first=True, padding_value = PAD_TOKEN)
        input_lengths = torch.LongTensor(n_of_words)
        
        if self.gpu:
            w_input = w_input.cuda()
            w_output = w_output.cuda()
            input_lengths = input_lengths.cuda()
        # print("utils.gen_subtensor正常")
        return w_input, w_output, input_lengths
        

    def collate_fun(self, data):
        # 四个形状为 [batch_size] 的长整型张量，分别表示一个 batch 中的正样本头实体 ID、正样本尾实体 ID、负样本头实体 ID 和负样本尾实体 ID
        # 四个形状为 [batch_size, max_ctx_len] 的长整型张量，分别表示一个 batch 中的正样本头实体上下文、正样本尾实体上下文、负样本头实体上下文和负样本尾实体上下文
        # 四个形状为 [batch_size] 的长整型张量，分别表示一个 batch 中的正样本头实体上下文长度、正样本尾实体上下文长度、负样本头实体上下文长度和负样本尾实体上下文长度
        # w_input, w_output两个形状为 [batch_size, max_seq_len] 的长整型张量，分别表示一个 batch 中的单词序列和单词序列标签
        # n_of_words 会被转换为一个形状为 [batch_size] 的长整型张量，表示一个 batch 中每个序列的长度
        # 执行dataloader的时候先执行 def __getitem__(self, idx) 返回的值在这里接收
        h_idx, t_idx, neg_h_idx, neg_t_idx, \
        h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, \
        neg_h_ctxs, neg_t_ctxs, neg_h_ctx_len, neg_t_ctx_len, \
        w_input, w_output, n_of_words, label = zip(*data) # loader中的属性， *号是解包运算符

        # print("h_idx:",len(h_idx)) # 32

# 先执行negs in neg_h_idx得到negs
        neg_h_idx = [neg for negs in neg_h_idx for neg in negs]
        neg_t_idx = [neg for negs in neg_t_idx for neg in negs]   
        hs = list(h_idx)+ list(neg_h_idx)
        ts = list(t_idx) + list(neg_t_idx)

        h_ctxs = [neg_ctx for negs in h_ctxs for neg_ctx in negs]
        t_ctxs = [neg_ctx for negs in t_ctxs for neg_ctx in negs]
        neg_h_ctxs = [neg_ctx for negs in neg_h_ctxs for neg_ctx in negs]
        neg_t_ctxs = [neg_ctx for negs in neg_t_ctxs for neg_ctx in negs]

        h_ctx_len = [neg_ctx for negs in h_ctx_len for neg_ctx in negs]
        t_ctx_len = [neg_ctx for negs in t_ctx_len for neg_ctx in negs]
        neg_h_ctx_len = [neg_ctx for negs in neg_h_ctx_len for neg_ctx in negs]
        neg_t_ctx_len = [neg_ctx for negs in neg_t_ctx_len for neg_ctx in negs]

        h_ctxs = h_ctxs + neg_h_ctxs 
        t_ctxs = t_ctxs + neg_t_ctxs
        h_ctx_len = h_ctx_len + neg_h_ctx_len
        t_ctx_len = t_ctx_len + neg_t_ctx_len

        # print("hs:",len(hs)) # 96
        label = label + (0,) * (len(hs) - len(label)) # 负样本标签用0填充
        # print("label:", len(label))  # 32

        # print("utils.collate_fun正常")
        return hs, ts, h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, w_input, w_output, n_of_words, label
    # hs:正负样本中所有的头部结点索引
    # ts：正负样本中所有的尾部结点索引
    # h_ctxs：头节点的上下文信息
    # t_ctxs：头节点的上下文信息
    # h_ctx_len, t_ctx_len： 上下文信息长度列表
    # w_input, w_output：一个batch中每对连接的单词序列表征和标签
    # n_of_words： 每对连接中边上文本内容的单词长度