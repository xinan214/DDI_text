import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

class HistDataset(Dataset):
    def __init__(self, loader, opt, train = True):
        self.df = loader.train_data
        self.word2idx = loader.word2idx
        self.G_out = loader.G_out
        self.G_in = loader.G_in
        self.n_ctxs = int(opt.n_ctxs) # 2
        self.neg = opt.neg
        self.init()
        if train:
            self.data = self.df.values 
        else:
            self.data = loader.test_data.values
        
       
    def ctx2words(self, context):
        return [self.word2idx[str(word)] for word in context.split()] # 通过 self.word2idx[str(word)] 来将每个单词转换为对应的索引值


    def init(self):
        self.heads = set(self.df['Drug A'])
        self.tails =set(self.df['Drug B'])
# self.head_ctx 和 self.tail_ctx，用于存储头部节点和尾部节点的上下文信息，
# 上下文信息：比如预测Vi与Vj是否右边及边上的文本 那么上下文查找头Vi和尾Vj的其他相连接的边上的信息，随机挑选n_ctx条作为上下文信息
        self.head_ctx = {}
        self.tail_ctx = {}
        for h in set(self.heads):
            # 遍历头部节点集合 self.heads，对于每个头部节点 h，代码通过访问有向图 self.G_out 中从头部节点 h 出发的边，提取每条边的上下文信息，并将这些上下文信息存储在 self.head_ctx[h] 字典中
            # self.G_out.edges([h], data=True) 获取从头部节点 h 出发的所有边，
            # 并遍历每条边的三元组 (source, target, data)，其中 data 是包含上下文信息的字典。将上下文信息 c['context'] 添加到 self.head_ctx[h] 字典中，形成一个列表
            self.head_ctx[h] = [c['DDI sentence'] for _,_,c in self.G_out.edges([h], data = True)]
  
        for t in set(self.tails):
            self.tail_ctx[t] = [c['DDI sentence'] for _,_,c in self.G_in.edges([t], data = True)]

        for i in range(1861): # 如果没有边上下文，填充
            if i not in self.head_ctx.keys():
                self.head_ctx[i] = ["PAD","PAD"]

            if i not in self.tail_ctx.keys():
                self.tail_ctx[i] = ["PAD","PAD"]



      

# 给定头部节点 h_id 和尾部节点 t_id，该方法会生成与它们不相关的负样本三元组
    def neg_triples(self, h_id, t_id):
        all_heads = self.heads
        all_tails = self.tails
        neg = []

   # 代码根据头部节点 h_id 在有向图 self.G_out 中的邻居节点，计算出不与头部节点 h_id 相连的尾部节点集合 ts
        ts = all_tails - set(self.G_out.neighbors(h_id))
        hs = all_heads - set(self.G_in.neighbors(t_id))
        
        # sample
        # 通过使用 np.random.randint(0, 2, self.neg) 生成长度为 self.neg 的随机整数数组，其中元素取值为 0 或 1
        chose = np.random.randint(0, 2, self.neg)
        h_neg = np.sum(chose == 0)
        t_neg = np.sum(chose == 1)
        # 从尾部节点集合 ts 中随机采样 h_neg 个尾部节点作为负样本的尾部节点集合 neg_tails
        neg_tails = random.sample(ts, h_neg)
        neg_heads = random.sample(hs, t_neg)

        # construct neg triples
        neg_h_id = neg_heads
        neg_t_id = [t_id]*len(neg_heads) # 原始尾部节点数扩充成和生成的负样本一样大的
        
        neg_t_id += neg_tails
        neg_h_id += [h_id]*len(neg_tails)
        return neg_h_id, neg_t_id
        
    
    def __len__(self):
        return len(self.data)
    
    
    def h_context(self, h_id):
        h_ctxs = np.random.choice(self.head_ctx[h_id], self.n_ctxs) # 从 self.head_ctx[h_id] 中随机选择 self.n_ctxs 个上下文信息（可以理解为与节点 h_id 相关的文本信息）
        h_ctxs = [self.ctx2words(ctx) for ctx in h_ctxs] # 将每个上下文信息转换为单词（word）的列表，即调用 self.ctx2words(ctx) 方法，其中 ctx 表示上下文信息
        return h_ctxs
    
    def t_context(self, t_id):
        if t_id not in self.tail_ctx.keys():
            print("tail_ctx.keys:",self.tail_ctx.keys())
            print(t_id)
        t_ctxs = np.random.choice(self.tail_ctx[t_id], self.n_ctxs)
        t_ctxs = [self.ctx2words(ctx) for ctx in t_ctxs]
        return t_ctxs
    
 # 我们通常使用 DataLoader 对象来加载数据集，DataLoader 对象会自动调用 __getitem__(self, idx) 函数来获取数据集中指定索引的元素
 # 假设 dataset 是一个 GraphDataset 对象，可以通过下标操作符访问其中的元素，如 dataset[0] 或 dataset[1]。这些操作会自动调用 __getitem__(self, idx) 函数，获取数据集中指定索引的元素
 # __getitem__(self, idx) 函数用于获取指定索引的数据，并将其转换为 PyTorch 中的张量对象。具体来说，该函数会根据输入的索引 idx，从 self.data 中获取对应的数据，然后将其转换为 PyTorch 中的张量对象，并返回这个张量对象
    def __getitem__(self, idx):
        h_id, t_id, clean_content, num_of_words, label = self.data[idx]
        content_idx = [self.word2idx[str(word)] for word in clean_content.split()]
        neg_h_id, neg_t_id = self.neg_triples(h_id, t_id)

        # print("neg_h_id:",len(neg_h_id)) # 2

        word_input = content_idx
        word_output = content_idx + [EOS_TOKEN]

        h_ctxs = self.h_context(h_id)
        t_ctxs = self.t_context(t_id)
        h_ctx_len = [len(h_ctx) for h_ctx in h_ctxs]
        t_ctx_len = [len(t_ctx) for t_ctx in t_ctxs]

        neg_h_ctxs = [ctx for neg_h in neg_h_id for ctx in self.h_context(neg_h)]
        neg_t_ctxs = [ctx for neg_t in neg_t_id for ctx in self.t_context(neg_t)]
        neg_h_ctx_len = [len(h_ctx) for h_ctx in neg_h_ctxs]
        neg_t_ctx_len = [len(t_ctx) for t_ctx in neg_t_ctxs]

        return h_id, t_id, neg_h_id, neg_t_id, \
            h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, \
            neg_h_ctxs, neg_t_ctxs, neg_h_ctx_len, neg_t_ctx_len, \
            word_input, word_output, num_of_words + 1, label

