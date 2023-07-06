# -*- coding: utf-8 -*-
'''main script of the project.

This script contains a single function that execute the contextual citation generation task.
The function is diveded into the following phases:
 - Loading processed the data.
 - model graph construction
 - model training
 - model evaluation
'''

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn import metrics 
from rouge import Rouge 
from tqdm import tqdm

from parser import *
from utils import *
from models import Decoder
from datasets import HistDataset

# train(model, train_dl, opt.epochs, optimizer)
def train(model, train_dl, epochs, optimizer):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.
        _step = 0
        with tqdm(total=len(train_dl), position=1, bar_format='{desc}') as desc:
            for batch in tqdm(train_dl, desc = '[Epoch {}]'.format(epoch+1), ncols = 80):
                hs, ts, out_ctx, in_ctx, h_lens, t_lens, w_input, w_output, n_of_words, label = [x for x in batch]

                # subgraph
                hs, ts, in_data, out_data = loader.graph_subtensor(hs, ts)
                in_ctx_data, out_ctx_data = loader.text_subtensor(in_ctx, t_lens, out_ctx, h_lens)
                # w_input, w_output两个形状为 [batch_size, max_seq_len] 的长整型张量，分别表示一个 batch 中的单词序列和单词序列标签
                w_input, w_output, input_lengths = loader.gen_subtensor(w_input, w_output, n_of_words) # 这个函数是为了把每个batch的序列长度填充为一致的
                #hs是边头顶点集合 ts是边尾顶点集合，in_data是内部图的信息，out_data是外部图的信息，in_ctx_data, out_ctx_data,是上下文信息
                # input_lengths表示一个 batch 中每个序列的长度
                scores, link_loss, gen_loss, step_loss = model(hs, ts,
                                                       in_data, out_data, 
                                                       in_ctx_data, out_ctx_data, 
                                                       w_input, w_output, input_lengths, label)
                
                # train
                model.zero_grad()
                step_loss.backward() # step_loss = link_loss + gen_loss：将链接损失和生成损失相加得到总损失
                # 梯度裁剪（gradient clipping）的函数，用于限制梯度的范数（norm）不超过给定的阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # 对梯度进行裁剪，防止梯度爆炸
                optimizer.step()  # 使用优化器更新模型参数
                epoch_loss += step_loss.cpu().detach()   # 累加每个批次的损失值，得到整个 epoch 的损失值
                del hs, ts, in_data, out_data, in_ctx_data, out_ctx_data, w_input, w_output, input_lengths
                
                _step += 1
                if _step % 10 == 1: # 每训练 10 个批次，输出一次当前的链接损失、生成损失和总损失
                    link_loss = round(link_loss.cpu().detach().item(), 2)
                    gen_loss = round(gen_loss.cpu().detach().item(), 2)
                    step_loss = round(step_loss.cpu().detach().item(), 2)
                    desc.set_description('[Train] #steps:{}\tgen:{}\tloss:{}'
                                         .format(link_loss, gen_loss, step_loss))
          
        desc.close()
        epoch_loss = round(epoch_loss.cpu().detach().item()/len(train_dl), 2)
        print('\nEpoch:{} \t loss:{} \n'.format(epoch, epoch_loss))
        return
  
    
def evaluation(model, test_dl, dataset):
    model.eval()
    total_auc = 0.
    rouge = Rouge()
    
    metrics_score = [[] for i in range(7)]
    generated_file = open('results/{}.txt'.format(dataset), "w") 
    
    _step = 0
    correct = 0
    test_loss = 0
    total = 0
    for batch in tqdm(test_dl, desc = '[evaluation]'):
        with torch.no_grad():
            hs, ts, out_ctx, in_ctx, h_lens, t_lens, w_input, w_output, n_of_words, label = [x for x in batch]
            print(len(label))

            # sub-graph
            hs, ts, in_data, out_data = loader.graph_subtensor(hs, ts)
            in_ctx_data, out_ctx_data = loader.text_subtensor(in_ctx, t_lens, out_ctx, h_lens)
            w_input, w_output, input_lengths = loader.gen_subtensor(w_input, w_output, n_of_words)  # 这个函数是为了把每个batch的序列长度填充为一致的
            link_head, link_tail, gen_head, gen_tail = model.evaluation_encoder(hs, ts,
                                                                                in_data, out_data,
                                                                                in_ctx_data, out_ctx_data)
            scores, link_loss, gen_loss, step_loss = model(hs, ts,
                                                   in_data, out_data,
                                                   in_ctx_data, out_ctx_data,
                                                   w_input, w_output, input_lengths, label)
            label = torch.tensor(label)
            # 计算损失和准确率
            test_loss += link_loss
            _, predicted = torch.max(scores.data, 1) # 选出概率最高的一个
            print("score.size:",scores.data.shape) # 96,114
            total += label.size(0) # 32
            print(predicted.shape)
            print(label.shape)
            correct += (predicted == label).sum().item() # label是一个数字


        # link prediction
        # head, tail = link_head, link_tail
        # pos_num = len(w_input)
        # neg_num = link_head.shape[0] - pos_num
        #
        # num_classes = 114  # 假设标签有114个类别
        # labels = torch.cat([torch.nn.functional.one_hot(label.long(), num_classes), torch.zeros(neg_num, num_classes)])
        # labels = torch.cat([torch.ones(pos_num), torch.zeros(neg_num)])
        # scores = torch.sigmoid(torch.sum(torch.mul(head, tail), dim=1))
        # auc = metrics.roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy(), average='weighted',
        #                             multi_class='ovr')
        # auc = metrics.roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
        # total_auc += auc

# ---------------------------------------------文本生成总是空串，评估暂不做-----------------------------------------------------
#         # context generation
#         head, tail = gen_head, gen_tail
#         pos_head = head[:pos_num,:]
#         pos_tail = tail[:pos_num,:]
#
#         for i in range(pos_num):
#             head = pos_head[i,:].unsqueeze(0)
#             tail = pos_tail[i,:].unsqueeze(0)
#             decoded_outputs = model.evaluate_decode(head, tail)
#             print("decoded_outputs:",decoded_outputs)
#             source_sentence = ' '.join([loader.idx2word[int(idx)] for idx in w_output[i].tolist()][:-1])
#             print("source_sentence:",source_sentence)
#             generated_sentence = ' '.join([loader.idx2word[idx] for idx in decoded_outputs])
#             print("generated_sentence:", generated_sentence)
#
# #             print('source_sentence:', source_sentence)
# #             print('generated_sentence:', generated_sentence)
# #             _step += 1
# #             generated_file.write("Case :" + str(_step) + "\n")
# #             generated_file.write('[source_sentence]\t' + source_sentence + "\n")
# #             generated_file.write('[generated_sentence]\t' + generated_sentence + "\n\n")
#
#             # 对于每个机器翻译的输出句子和对应的参考翻译，分别计算 1-gram、2-gram、3-gram 和 4-gram 的重叠程度。这里的 n-gram 指的是包含 n 个连续词的词组
#
#             # BLEU
#             # BLEU 指标计算机器翻译结果中 n-gram 词组与参考翻译中的 n-gram 词组之间的重叠程度，然后将重叠程度加权平均得到 BLEU 分数。
#             # BLEU 指标的取值范围在 0 到 1 之间，分数越高表示机器翻译结果越接近参考翻译
#             bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([source_sentence], [generated_sentence])
#             metrics_score[0].append(bleu_1)
#             metrics_score[1].append(bleu_2)
#             metrics_score[2].append(bleu_3)
#             metrics_score[3].append(bleu_4)
#
#             # ROUGE
#             # 用于衡量机器生成的摘要与参考摘要之间的相似度。
#             # ROUGE 指标计算机器生成的摘要和参考摘要之间的共同 n-gram 词组的数量，然后根据不同的 ROUGE 指标，将共同词组数量加权平均得到 ROUGE 分数。
#             # ROUGE 指标的取值范围在 0 到 1 之间，分数越高表示机器生成的摘要越接近参考摘要
#             rouge_1 = rouge.get_scores(generated_sentence, source_sentence)
#             metrics_score[4].append(rouge_1[0]['rouge-1']['f'])
#             metrics_score[5].append(rouge_1[0]['rouge-2']['f'])
#             metrics_score[6].append(rouge_1[0]['rouge-l']['f'])

        # 计算平均损失和准确率
        del hs, ts, in_data, out_data, in_ctx_data, out_ctx_data
    # 计算平均损失和准确率

    avg_loss = test_loss / total
    accuracy = 100 * correct / total

    print("test_avg_loss:{:.3f},test_accuracy:{:.3f}".format(avg_loss, accuracy))
    # auc = total_auc/len(test_dl)    # 计算所有样本平均AOC
    # metrics_score = [sum(i)/len(i) for i in metrics_score]
    # metrics_score 是一个长度为 7 的列表，其中前 4 个元素保存的是 BLEU-1、BLEU-2、BLEU-3 和 BLEU-4 四个指标的值，后 3 个元素保存的是 ROUGE-1、ROUGE-2 和 ROUGE-L 三个指标的值。
    # 因此，使用 metrics_score[:4] 可以提取出前 4 个元素，即 BLEU 指标的值，保存到 bleu_score 列表中；使用 metrics_score[4:7] 可以提取出后 3 个元素，即 ROUGE 指标的值，
    # 保存到 rouge_score 列表中
    # bleu_score = metrics_score[:4]
    # rouge_score = metrics_score[4:7]
    return auc #, bleu_score, rouge_score



if __name__ == '__main__':   
    #---------------- Loading data phase -----------------------
    opt = get_parser()
    loader = Loader(opt.dataset, opt.gpu)  # 返回的是一个 Loader 类的实例，有loader self的各种属性

  # train_data:
  #   paper_id1     paper_id2   clean_content   num_of_words
    model = Decoder(n_words=loader.n_words,  # 9860
                    n_nodes=loader.n_nodes,  # 101
                    max_len=loader.max_len,  # 191
                    opt = opt)
    if opt.gpu: model= model.cuda()

    # Adam 优化器适合处理稀疏梯度和非平稳目标函数
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # Adagrad 优化器适合处理具有稀疏梯度和欠约束的目标函数
    elif opt.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt.lr)
    # SGD 优化器则更适合处理大规模数据集和稳定的目标函数
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
 

    # ---------------------------- Training phase ----------------------------------
    # print('Training...')
    # train_dataset = HistDataset(loader, opt) # 数据集相关 负样本 划分等
    # # collate_fn = loader.collate_fun表示使用loader对象的collate_fun方法来对每个batch中的数据进行处理和组合。
    # # collate_fn方法的作用是将一个batch中的数据转换为模型可以直接处理的格式，例如将输入序列填充到相同的长度、将标签转换为one - hot编码等
    # # collate_fun方法的作用是将每个样本中的特征和信息组合成一个批次数据，在这个过程中会自动从数据集中生成正负样本
    # # 我们通常使用 DataLoader 对象来加载数据集，DataLoader 对象会自动调用 __getitem__(self, idx) 函数来获取数据集中指定索引的元素
    # train_dl = DataLoader(train_dataset,
    #                       opt.batch_size,
    #                       pin_memory=True,
    #                       shuffle=True,
    #                       collate_fn=loader.collate_fun,
    #                       num_workers=1)
    # train(model, train_dl, opt.epochs, optimizer)
    #
    # # 保存模型
    # torch.save(model.state_dict(), './results/model.pth')
    # print("----------------------------------------train over!-------------------------------------------")

    #---------------------------- Evaluation phase ----------------------------
    print('Evaluation...')
    # 加载模型
    model.load_state_dict(torch.load('./results/model.pth'))
    test_dataset = HistDataset(loader, opt, False)
    test_dl = DataLoader(test_dataset,
                         opt.batch_size,
                         pin_memory=True,
                         collate_fn=loader.collate_fun)
    auc = evaluation(model, test_dl, opt.dataset)

    print('AUC:{:.3f}'.format(auc))
    # print('BLEU:', bleu_score)
    # print('ROUGE:', rouge_score)

    # ---------------------------------------------------------------------------
